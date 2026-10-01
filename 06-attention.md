# 第 6 章 Attention：KV Cache 的读写与 FlashAttention

> 本章目标：理解 KV Cache 在物理上到底怎么写进去、attention 到底怎么读出来。
> 这是把第 4 章的"块表"和第 5 章的"slot_mapping"落到实处的最后一环。

## 6.1 本章要回答的三个问题

1. 新算出来的 K/V，**写到 KV Cache 的哪个位置**？（Triton kernel）
2. prefill 时注意力怎么算？为什么用 varlen？
3. decode 时注意力怎么算？为什么要从 cache 里读？

## 6.2 全章代码

`nanovllm/layers/attention.py` 只有 75 行，但信息密度极高：

```python
import torch
from torch import nn
import triton
import triton.language as tl

from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache
from nanovllm.utils.context import get_context


@triton.jit
def store_kvcache_kernel(
    key_ptr, key_stride, value_ptr, value_stride,
    k_cache_ptr, v_cache_ptr, slot_mapping_ptr, D: tl.constexpr,
):
    idx = tl.program_id(0)
    slot = tl.load(slot_mapping_ptr + idx)
    if slot == -1: return
    key_offsets = idx * key_stride + tl.arange(0, D)
    value_offsets = idx * value_stride + tl.arange(0, D)
    key = tl.load(key_ptr + key_offsets)
    value = tl.load(value_ptr + value_offsets)
    cache_offsets = slot * D + tl.arange(0, D)
    tl.store(k_cache_ptr + cache_offsets, key)
    tl.store(v_cache_ptr + cache_offsets, value)


def store_kvcache(key, value, k_cache, v_cache, slot_mapping):
    N, num_heads, head_dim = key.shape
    D = num_heads * head_dim
    assert key.stride(-1) == 1 and value.stride(-1) == 1
    assert key.stride(1) == head_dim and value.stride(1) == head_dim
    assert k_cache.stride(1) == D and v_cache.stride(1) == D
    assert slot_mapping.numel() == N
    store_kvcache_kernel[(N,)](key, key.stride(0), value, value.stride(0), k_cache, v_cache, slot_mapping, D)


class Attention(nn.Module):
    def __init__(self, num_heads, head_dim, scale, num_kv_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.k_cache = self.v_cache = torch.tensor([])

    def forward(self, q, k, v):
        context = get_context()
        k_cache, v_cache = self.k_cache, self.v_cache
        if k_cache.numel() and v_cache.numel():
            store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)
        if context.is_prefill:
            if context.block_tables is not None:    # prefix cache
                k, v = k_cache, v_cache
            o = flash_attn_varlen_func(q, k, v,
                                       max_seqlen_q=context.max_seqlen_q, cu_seqlens_q=context.cu_seqlens_q,
                                       max_seqlen_k=context.max_seqlen_k, cu_seqlens_k=context.cu_seqlens_k,
                                       softmax_scale=self.scale, causal=True, block_table=context.block_tables)
        else:    # decode
            o = flash_attn_with_kvcache(q.unsqueeze(1), k_cache, v_cache,
                                        cache_seqlens=context.context_lens, block_table=context.block_tables,
                                        softmax_scale=self.scale, causal=True)
        return o
```

## 6.3 KV Cache 的物理布局

在讲 kernel 之前，先明确 KV Cache 长什么样（第 5 章分配的那个大 tensor）：

```
self.kv_cache: [2, num_layers, num_blocks, block_size, num_kv_heads, head_dim]
                              ↑           ↑
                         物理块号     块内位置
```

每一层拿到的切片 `k_cache` 形状是 `[num_blocks, block_size, num_kv_heads, head_dim]`。

### 关键：把它看成二维的"槽位空间"

对于写入，我们只关心前两维：
```
k_cache 看成 [num_blocks × block_size, num_kv_heads × head_dim]
              ↑                          ↑
         "槽位号" slot            每个槽位存 D 个数
```

**槽位号的计算**：`slot = block_id × block_size + 块内偏移`

这就是为什么第 5 章的 `slot_mapping` 长那样：
- prefill：`seq.block_table[i] * block_size + offset`
- decode：`seq.block_table[-1] * block_size + last_block_num_tokens - 1`

### 一个具体例子

block_size=4（简化），num_kv_heads=2，head_dim=3：D = 2×3 = 6

```
物理块 0: 槽0  槽1  槽2  槽3     (每槽 6 个数)
物理块 1: 槽4  槽5  槽6  槽7
物理块 2: 槽8  槽9  槽10 槽11
```

序列 A 的 block_table = [2, 0]：
- 它的第 0 个 token 的 KV 存在**槽 8**（块2的第0位）
- 它的第 1 个 token 存在**槽 9**
- ...
- 它的第 4 个 token 存在**槽 0**（块0的第0位）—— 跳跃到另一个物理块

**逻辑连续，物理分散**——这就是分页。

## 6.4 Triton kernel：`store_kvcache_kernel`

### 先理解 Triton 的心智模型

Triton 让你用 Python 写 GPU kernel。核心概念：

- **`tl.program_id(0)`**：当前"程序实例"的编号。
  启动时 `store_kvcache_kernel[(N,)]` 表示启动 N 个实例。
- **每个实例处理一块数据**（这里是第 idx 个 token）。
- **`tl.arange(0, D)`**：生成向量 `[0,1,2,...,D-1]`。
- **指针 + 向量**：`ptr + arange` 表示一组连续地址，可向量化读写。

### 逐行解读

```python
idx = tl.program_id(0)
```
第 idx 个实例，负责处理扁平化后的第 idx 个 token。

```python
slot = tl.load(slot_mapping_ptr + idx)
if slot == -1: return
```
读槽位号。⚠️ `slot == -1` 是 **padding**（比如 CUDA Graph 里的无效项），
直接返回不写。

> 注意 Triton 里 `if` 是"带条件的返回"，整个 program 实例提前退出，
> 不是真的分支控制流。

```python
key_offsets = idx * key_stride + tl.arange(0, D)
value_offsets = idx * value_stride + tl.arange(0, D)
```
计算当前 token 的 K/V 在**输入张量**里的地址。`key_stride` 是行步长
（一个 token 占多少个数），`arange(0, D)` 是行内偏移。

```python
key = tl.load(key_ptr + key_offsets)
value = tl.load(value_ptr + value_offsets)
```
按基址 + 偏移读入 K/V 各一行（D 个数）。

```python
cache_offsets = slot * D + tl.arange(0, D)
tl.store(k_cache_ptr + cache_offsets, key)
tl.store(v_cache_ptr + cache_offsets, value)
```
写入 cache 的 `slot` 位置。

⚠️ **注意地址公式 `slot * D`**：这就是"槽位号 → 物理地址"的乘法。
因为每层的 k_cache 是 `[num_blocks, block_size, num_kv_heads, head_dim]`
的连续张量，展平后 `[num_blocks*block_size, D]`，
槽位间步长正好是 D。**一次乘法搞定地址换算，无需查表。**

### host 端的 wrapper：`store_kvcache`

```python
def store_kvcache(key, value, k_cache, v_cache, slot_mapping):
    N, num_heads, head_dim = key.shape
    D = num_heads * head_dim
    assert key.stride(-1) == 1 and value.stride(-1) == 1
    assert key.stride(1) == head_dim and value.stride(1) == head_dim
    assert k_cache.stride(1) == D and v_cache.stride(1) == D
    assert slot_mapping.numel() == N
    store_kvcache_kernel[(N,)](key, key.stride(0), value, value.stride(0), k_cache, v_cache, slot_mapping, D)
```

### 那堆 assert 是干什么的？

Triton kernel 里假设数据是**紧密排布**的（地址能用简单乘加算出来）。
assert 保证这个假设成立：

```python
assert key.stride(-1) == 1          # 最后一个维度连续
assert key.stride(1) == head_dim    # head 维连续（头之间无 padding）
assert k_cache.stride(1) == D       # cache 的 head 维也是连续的
```

`key` 形状是 `[N, num_heads, head_dim]`：
- `stride(-1) == 1`：head_dim 连续
- `stride(1) == head_dim`：说明 `[num_heads, head_dim]` 整体连续，可看成 `[num_heads*head_dim]`

如果这些不成立（比如 key 是某个切片的转置），地址算错会写坏内存。
assert 是廉价的安全网。

### 为什么用 Triton 而不是 PyTorch 索引？

可以这么写（纯 PyTorch）：
```python
k_cache[slot_mapping] = key   # 散射赋值
```

但 `slot_mapping` 是**任意的**（块可能不连续），
PyTorch 的高级索引会做 gather/scatter，速度慢且可能额外分配临时张量。
Triton kernel 直接控制地址，**一次写准**，没有额外开销。

### 启动配置 `(N,)`

```python
store_kvcache_kernel[(N,)](...)
```
启动 N 个实例（N = token 数）。每个实例处理一个 token。
对于 prefill 的长序列，N 可能上千，并行度很好。

## 6.5 `Attention` 模块

### 初始化

`attention.py:43`：

```python
class Attention(nn.Module):
    def __init__(self, num_heads, head_dim, scale, num_kv_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.k_cache = self.v_cache = torch.tensor([])   # 占位
```

⚠️ **`self.k_cache = self.v_cache = torch.tensor([])`** 是占位。

为什么？因为 `Attention` 构造时 KV Cache 还没分配（第 5 章里
`allocate_kv_cache` 在模型构造**之后**）。先用空张量占位，
好让 `allocate_kv_cache` 通过 `hasattr(module, "k_cache")` 找到并替换。

这也让 `forward` 里可以判断：
```python
if k_cache.numel() and v_cache.numel():   # 空张量 numel()=0，跳过
```
warmup 时 cache 还是空的，就不写 KV。

### `forward` 的三段式

`attention.py:59`：

```python
def forward(self, q, k, v):
    context = get_context()
    k_cache, v_cache = self.k_cache, self.v_cache

    # 第1段：写 KV Cache
    if k_cache.numel() and v_cache.numel():
        store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)

    # 第2段：prefill
    if context.is_prefill:
        if context.block_tables is not None:    # prefix cache
            k, v = k_cache, v_cache
        o = flash_attn_varlen_func(q, k, v, ...)
    # 第3段：decode
    else:
        o = flash_attn_with_kvcache(q.unsqueeze(1), k_cache, v_cache, ...)

    return o
```

**注意第1段是无条件的**（只要 cache 存在就写）。
无论 prefill 还是 decode，本步算出的新 KV 都要落盘。
然后第2、3段用不同的方式做注意力。

## 6.6 第1段：写 KV Cache

```python
if k_cache.numel() and v_cache.numel():
    store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)
```

对 prefill 和 decode 是同一套逻辑：按 `slot_mapping` 写。
不同的是 `slot_mapping` 的内容（第 5 章讲过）：
- prefill：一堆连续但跨块的槽位
- decode：每个序列 1 个槽位

**为什么先写再算？** 因为 attention 要读到刚写入的当前 token 的 KV。
如果顺序反了，当前 token 会读不到自己的 KV。

## 6.7 第2段：Prefill 的 attention

```python
if context.block_tables is not None:    # prefix cache
    k, v = k_cache, v_cache
o = flash_attn_varlen_func(q, k, v,
                           max_seqlen_q=context.max_seqlen_q, cu_seqlens_q=context.cu_seqlens_q,
                           max_seqlen_k=context.max_seqlen_k, cu_seqlens_k=context.cu_seqlens_k,
                           softmax_scale=self.scale, causal=True, block_table=context.block_tables)
```

### 两个分支

**(a) 普通 prefill**（无前缀缓存）：`block_tables is None`

直接用计算出的 `k, v`（本步算的那些新 token 的 KV）做 attention。
因为所有 token 都在这一次算出来，K/V 张量已经包含了需要的一切。

**(b) 带前缀缓存的 prefill**：`block_tables is not None`

```python
k, v = k_cache, v_cache
```

⚠️ **关键替换**：把 k、v 换成**整个 KV Cache**！

为什么？因为前缀部分的 KV 是之前算好存在 cache 里的，不在本步的 `k, v` 里。
attention 需要访问全部历史 KV，所以要直接从 cache 读。

那怎么从 cache 里只读属于当前序列的那部分？——靠 `block_table` 参数。
flash-attn 会根据 block_table 找到该序列占用的物理块，
再结合 `cu_seqlens_k` 知道每个序列的 key 长度。

### `flash_attn_varlen_func` 参数解读

```python
o = flash_attn_varlen_func(
    q, k, v,
    max_seqlen_q=..., cu_seqlens_q=...,   # query 的变长信息
    max_seqlen_k=..., cu_seqlens_k=...,   # key/value 的变长信息
    softmax_scale=self.scale,             # 1/sqrt(head_dim)
    causal=True,                          # 因果掩码（只能看前面）
    block_table=context.block_tables,     # 分页定位（可选）
)
```

| 参数 | 作用 |
|------|------|
| `q, k, v` | 扁平拼接的变长张量 |
| `cu_seqlens_q/k` | 每个序列在扁平张量里的边界 |
| `max_seqlen_q/k` | 最长序列长度（flash-attn 用来分配 block） |
| `causal=True` | 因果注意力 |
| `block_table` | 分页模式下 KV 的物理位置 |

### 为什么 q 和 k 的长度可能不同？

回顾第 5 章：前缀缓存时 `seqlen_k > seqlen_q`。

```
序列 A: prompt 600 + 命中缓存 512
  q 长度 = 88（只算新 token 的 query）
  k 长度 = 600（attention 要看全部 600 个 key）
```

所以 `cu_seqlens_q = [0, 88, ...]`，
`cu_seqlens_k = [0, 600, ...]`。

flash-attn 需要支持这种 "q 短 k 长" 的非对称情况 —— 这正是
`block_table` 参数存在的意义（从 cache 读长的 k）。

## 6.8 第3段：Decode 的 attention

```python
else:    # decode
    o = flash_attn_with_kvcache(q.unsqueeze(1), k_cache, v_cache,
                                cache_seqlens=context.context_lens, block_table=context.block_tables,
                                softmax_scale=self.scale, causal=True)
```

### 为什么用不同的函数？

decode 的形态和 prefill 完全不同：
- **q**：每个序列只有 1 个 token（当前生成的）
- **k/v**：在 KV Cache 里（历史全部）

所以用 `flash_attn_with_kvcache` —— 专门为"query 短、KV 在 cache 里"
的场景优化。

### 参数解读

| 参数 | 作用 |
|------|------|
| `q.unsqueeze(1)` | 给 q 加一个 seqlen 维度（从 `[bs, h, d]` 变 `[bs, 1, h, d]`） |
| `k_cache, v_cache` | 分页 KV Cache |
| `cache_seqlens` | 每个序列的上下文长度（告诉 flash-attn 读多少） |
| `block_table` | 物理块定位 |

### `q.unsqueeze(1)` 的作用

flash-attn 的 `with_kvcache` 期望 q 形如 `[batch, seqlen, num_heads, head_dim]`。
decode 时 seqlen=1，所以 `unsqueeze(1)` 把 `[bs, h, d]` 变 `[bs, 1, h, d]`。

### `cache_seqlens`（即 context_lens）

告诉 flash-attn 每个序列在 cache 里有多少个有效的 KV。
例：序列长度 601 → cache_seqlens=601，从 block_table 指定的块里读 601 个 KV。

⚠️ 这个值必须精确，否则会读到垃圾（未初始化的 cache 位置）。

## 6.9 完整实例：一次 prefill + 两次 decode

设 block_size=4（简化），序列 A 的 prompt = 5 个 token，
block_table = [2, 0]。KV Cache 是 `[8 槽位, D]`。

### Prefill

`num_cached_tokens=0, num_scheduled_tokens=5`。

`prepare_prefill` 算出的 slot_mapping：
- start=0, end=5
- start_block = 0//4 = 0, end_block = (5+3)//4 = 2
- 块0（block_table[0]=2）：
  - i==start_block 且 i!=end_block-1 → 整块
  - slot_start = 2*4 + 0 = 8，slot_end = 2*4+4 = 12
  - → `[8, 9, 10, 11]`
- 块1（block_table[1]=0）：
  - i==end_block-1 → `slot_end = 0*4 + 5 - 1*4 = 1`
  - i!=start_block → slot_start = 0*4 = 0
  - → `[0]`
- 总计 `slot_mapping = [8, 9, 10, 11, 0]`

**含义**：5 个 token 的 KV 中，前 4 个写到块2（槽8-11），
第 5 个写到块0（槽0）。

```
KV Cache:
槽位:  0    1    2    3  | 4 5 6 7 | 8    9    10   11
内容: A4   -    -    -  | - - - - | A0   A1   A2   A3
```

`store_kvcache` 按这个映射写入。然后 `flash_attn_varlen_func` 算 attention
（此时 block_tables 为 None，因为无前缀缓存）。

采样得到 token A5，`num_tokens` 变成 6。

### Decode 第1次

输入 A5，`num_tokens = 6`。

`prepare_decode` 算 slot_mapping：
- `block_table[-1] = 0`，`last_block_num_tokens = 6 - 1*4 = 2`
- slot = `0*4 + 2 - 1 = 1`

**含义**：A5 的 KV 写到槽 1（块0的第2个位置）。

```
槽位:  0    1    2    3  | ...
内容: A4   A5   -    -  | ...
```

`block_tables = [[2, 0]]`，`context_lens = [6]`。

`flash_attn_with_kvcache` 从块2（槽8-11 存 A0-A3）、块0（槽0-1 存 A4-A5）
读出 6 个 KV，和 q（A5）做 attention。

采样得到 A6，`num_tokens = 7`。

### Decode 第2次

输入 A6，`num_tokens = 7`。
- `last_block_num_tokens = 7 - 4 = 3`，slot = `0*4 + 3 - 1 = 2`
- 写入槽 2

```
槽位:  0    1    2    3  | ...
内容: A4   A5   A6   -  | ...
```

继续。

### 观察

- prefill 一次写 5 个 KV（跨 2 个块）
- decode 每次写 1 个 KV，直到块0满了（槽3），下次 decode 会触发新块分配
  （`may_append` 的 `len(seq) % block_size == 1` 条件）

## 6.10 为什么 attention 层要"感知分页"？

一个可能的疑惑：为什么不把 KV Cache 当成简单的连续数组，
用普通的 attention 就好了？

因为**分页是为了显存效率**（第 4 章）。如果 KV 是连续的，
就没法共享前缀、没法按块回收、没法消除碎片。

代价就是 attention 必须能"跨块读取"，这就需要 flash-attn 的
`block_table` 参数支持。所以 attention 层必须和下层的分页管理
**协同设计**——这正是 PagedAttention 论文的核心贡献。

## 6.11 Prefill vs Decode 全对比

| 维度 | Prefill | Decode |
|------|---------|--------|
| q 的形状 | N 个 token（变长） | 每序列 1 个 token |
| k/v 来源 | 本步算的（+cache 若有前缀） | KV Cache（`flash_attn_with_kvcache`） |
| flash-attn 函数 | `flash_attn_varlen_func` | `flash_attn_with_kvcache` |
| 关键参数 | `cu_seqlens_q/k`, `max_seqlen` | `cache_seqlens (context_lens)` |
| 是否需要 block_table | 仅前缀缓存时需要 | 总是需要 |
| 写入 KV 数量 | 本步所有新 token | 每序列 1 个 |
| CUDA Graph | 不用（形状动态） | 用（batch 档位） |

## 6.12 本章小结

- KV Cache 物理上是 `[num_blocks, block_size, kv_heads, head_dim]`，
  展平后可看成 `[槽位, D]`，槽位号 = `block_id × block_size + 偏移`。
- **`store_kvcache`** 用 Triton kernel 按 `slot_mapping` 散射写入，
  一次乘法算地址，无查表。
- Triton kernel 假设数据紧密排布，故 host 端有一堆 stride 断言。
- `Attention.forward` 三段式：**先写盘（无条件）→ 再按 prefill/decode 分支算**。
- prefill 用 `flash_attn_varlen_func`（变长拼接 + cu_seqlens 边界）；
  有前缀缓存时 `k, v` 直接换成整个 cache（靠 block_table 定位）。
- decode 用 `flash_attn_with_kvcache`，q 只有一个 token，
  `cache_seqlens` 告诉它读多少历史 KV。
- 分页是显存效率的代价，attention 必须感知 block_table 才能跨块读取。

**下一章**：[模型与张量并行](07-model-tp.md)——Qwen3 的结构和各层的切分方式。
