# 第 7 章 模型与张量并行

> 本章目标：理解 Qwen3 的模型结构如何用最少的代码实现，
> 以及四种线性层如何实现张量并行（Tensor Parallel）。

## 7.1 张量并行要解决什么

一个 70B 参数的模型放不下一张卡，怎么办？**把权重切开，分散到多张卡**，
每张卡算一部分，最后汇合。这就是张量并行（TP）。

Nano-vLLM 支持 `tensor_parallel_size ∈ [1, 8]`。核心思想：

```
一张卡:   Y = X · W            (W 是完整权重矩阵)

两张卡:   W = [W_top]          Y_top    = X · W_top
              [W_bot]          Y_bot    = X · W_bot
                                Y = concat(Y_top, Y_bot)     (列并行)

或者:     W = [W_left | W_right]   Y = X_left·W_left + X_right·W_right  (行并行)
```

关键观察：**切分只有在特定位置才"通信免费"**。
一个 MLP 是 `down(silu(gate(x)) * up(x))`，
如果 `gate`、`up` 按列切（输出切），那 silu/mul 可以本地做，
但 `down` 需要完整输入 —— 所以 `down` 按行切（输入切），
结果做 `all_reduce` 汇总。

**规则**：Transformer 里 `Linear → 激活 → Linear` 的模式，
前一个用列并行（Column Parallel），后一个用行并行（Row Parallel），
这样**每个 block 只需一次 all_reduce**。

## 7.2 四种线性层总览

`nanovllm/layers/linear.py` 定义了四种：

| 层 | 切分方式 | 通信 | 用在 |
|----|---------|------|------|
| `ReplicatedLinear` | 不切，各卡完整副本 | 无 | （本模型未用） |
| `ColumnParallelLinear` | 按输出维切 | 无 | 前半个 MLP |
| `RowParallelLinear` | 按输入维切 | `all_reduce` | 后半个 MLP、o_proj |
| `MergedColumnParallelLinear` | 多个输出合并后切 | 无 | gate+up 合并 |
| `QKVParallelLinear` | Q/K/V 合并后切 | 无 | QKV 投影 |

### 基类 `LinearBase`

`linear.py:12`：

```python
class LinearBase(nn.Module):

    def __init__(self, input_size, output_size, bias=False, tp_dim=None):
        super().__init__()
        self.tp_dim = tp_dim
        self.tp_rank = dist.get_rank()
        self.tp_size = dist.get_world_size()
        self.weight = nn.Parameter(torch.empty(output_size, input_size))
        self.weight.weight_loader = self.weight_loader
        if bias:
            self.bias = nn.Parameter(torch.empty(output_size))
            self.bias.weight_loader = self.weight_loader
        else:
            self.register_parameter("bias", None)
```

要点：
- 记录 `tp_rank`（我是哪张卡）、`tp_size`（几张卡）
- `tp_dim`：在哪个维度切（0=输出/列，1=输入/行）
- **`self.weight.weight_loader = self.weight_loader`**：
  把加载函数挂到参数上。这样 `utils/loader.py` 加载权重时可以直接调用，
  实现"边加载边切分"。这是这套设计的精髓。
- `register_parameter("bias", None)`：显式声明无 bias
  （而不是不创建），能让 `state_dict` 保持一致

### `divide` 辅助函数

`linear.py:7`：

```python
def divide(numerator, denominator):
    assert numerator % denominator == 0
    return numerator // denominator
```

断言能整除。TP 要求所有维度能被 `tp_size` 整除，否则报错。

## 7.3 `ColumnParallelLinear`：按输出切

`linear.py:54`：

```python
class ColumnParallelLinear(LinearBase):

    def __init__(self, input_size, output_size, bias=False):
        tp_size = dist.get_world_size()
        super().__init__(input_size, divide(output_size, tp_size), bias, 0)

    def weight_loader(self, param, loaded_weight):
        param_data = param.data
        shard_size = param_data.size(self.tp_dim)
        start_idx = self.tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(self.tp_dim, start_idx, shard_size)
        param_data.copy_(loaded_weight)

    def forward(self, x):
        return F.linear(x, self.weight, self.bias)
```

### 形状变化

```
完整权重: W [output_size, input_size]
每张卡:  W_shard [output_size/tp_size, input_size]
         (tp_dim=0，在第0维切)
```

### `weight_loader`：切分加载

从 HF 加载完整权重 `loaded_weight`，只取自己那一片：

```python
shard_size = param_data.size(self.tp_dim)      # 我该有多大
start_idx = self.tp_rank * shard_size          # 我这一片的起点
loaded_weight = loaded_weight.narrow(self.tp_dim, start_idx, shard_size)
param_data.copy_(loaded_weight)
```

`narrow(dim, start, length)` = 取切片。

**例**：output_size=4096, tp_size=2, rank=1
- 每卡权重 `[2048, input_size]`
- `shard_size=2048`，`start_idx = 1*2048 = 2048`
- 取 `loaded_weight[2048:4096]` ✓

### 为什么列并行不需要通信？

```python
def forward(self, x):
    return F.linear(x, self.weight, self.bias)
```

输入 `x` 是完整的（**所有卡都有完整的 x**），输出被切成两半。
每张卡输出自己那半，**不需要通信**。通信留给后面的行并行层。

## 7.4 `RowParallelLinear`：按输入切 + all_reduce

`linear.py:131`：

```python
class RowParallelLinear(LinearBase):

    def __init__(self, input_size, output_size, bias=False):
        tp_size = dist.get_world_size()
        super().__init__(divide(input_size, tp_size), output_size, bias, 1)

    def weight_loader(self, param, loaded_weight):
        param_data = param.data
        if param_data.ndim == 1:
            param_data.copy_(loaded_weight)
            return
        shard_size = param_data.size(self.tp_dim)
        start_idx = self.tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(self.tp_dim, start_idx, shard_size)
        param_data.copy_(loaded_weight)

    def forward(self, x):
        y = F.linear(x, self.weight, self.bias if self.tp_rank == 0 else None)
        if self.tp_size > 1:
            dist.all_reduce(y)
        return y
```

### 形状变化

```
完整权重: W [output_size, input_size]
每张卡:  W_shard [output_size, input_size/tp_size]
         (tp_dim=1，在第1维切)
```

### `forward` 的三步

```python
y = F.linear(x, self.weight, self.bias if self.tp_rank == 0 else None)
```

⚠️ **bias 只在 rank 0 加**！为什么？

因为每张卡算出的是**部分和**，最后要 `all_reduce`（求和）。
如果每张卡都加 bias，求和后 bias 就被加了 `tp_size` 次，变成 `tp_size × bias`。
所以只让 rank 0 加一次。

```python
if self.tp_size > 1:
    dist.all_reduce(y)
```

把各卡的部分和汇总。`all_reduce` 默认是求和。

### 为什么这里需要通信？

因为输出是完整的（`[batch, output_size]`），但每张卡只有部分输入算出的部分和。
必须求和才能得到正确结果。

### `weight_loader` 的特判

```python
if param_data.ndim == 1:
    param_data.copy_(loaded_weight)
    return
```

一维参数（比如 bias）不切分，直接拷贝。
因为 bias 是 `[output_size]`，行并行不切输出维。

## 7.5 `MergedColumnParallelLinear`：合并多个列并行权重

`linear.py:76`：

```python
class MergedColumnParallelLinear(ColumnParallelLinear):

    def __init__(self, input_size, output_sizes, bias=False):
        self.output_sizes = output_sizes
        super().__init__(input_size, sum(output_sizes), bias)

    def weight_loader(self, param, loaded_weight, loaded_shard_id):
        param_data = param.data
        shard_offset = sum(self.output_sizes[:loaded_shard_id]) // self.tp_size
        shard_size = self.output_sizes[loaded_shard_id] // self.tp_size
        param_data = param_data.narrow(self.tp_dim, shard_offset, shard_size)
        loaded_weight = loaded_weight.chunk(self.tp_size, self.tp_dim)[self.tp_rank]
        param_data.copy_(loaded_weight)
```

### 为什么合并？

MLP 有 `gate_proj` 和 `up_proj` 两个独立的矩阵乘：
```python
gate = x @ W_gate.T
up   = x @ W_up.T
out  = silu(gate) * up
```

两次矩阵乘意味着两次 kernel 启动。**合并成一个**矩阵乘
（权重 `[W_gate; W_up]` 拼接），kernel 启动减半，大矩阵乘效率更高：

```python
gate_up = x @ [W_gate; W_up].T    # 一次算完
gate, up = gate_up.chunk(2, -1)
```

### `weight_loader` 的切片逻辑

HF 模型里 `gate_proj` 和 `up_proj` 是分开存储的。
加载时要分别取它们、切分、放进合并权重的对应位置。

```python
shard_offset = sum(self.output_sizes[:loaded_shard_id]) // self.tp_size
shard_size = self.output_sizes[loaded_shard_id] // self.tp_size
```

- `loaded_shard_id = 0`（gate）：`shard_offset = sum([])//tp = 0`
- `loaded_shard_id = 1`（up）：`shard_offset = sum([intermediate])//tp`

⚠️ **注意 `shard_offset` 要除以 `tp_size`**！因为本地权重已经是切分后的，
偏移也必须在切分后的坐标系里。

### 一个例子

假设 `intermediate_size = 4096`, `tp_size = 2`：

```
本地合并权重形状 [4096, hidden]   (sum(output_sizes)//tp = 8192/2)
本地布局: [gate_shard(2048) | up_shard(2048)]

加载 gate (shard_id=0):
  shard_offset = 0//2 = 0, shard_size = 4096//2 = 2048
  取完整 gate 权重的第 rank 片 → 放到本地偏移 0

加载 up (shard_id=1):
  shard_offset = 4096//2 = 2048, shard_size = 2048
  取完整 up 权重的第 rank 片 → 放到本地偏移 2048
```

注意 `loaded_weight.chunk(tp_size, tp_dim)[tp_rank]`：HF 里 gate/up 是**完整**的，
要先 chunk 取自己那片。

## 7.6 `QKVParallelLinear`：Q/K/V 合并切分

`linear.py:96`：

```python
class QKVParallelLinear(ColumnParallelLinear):

    def __init__(self, hidden_size, head_size, total_num_heads, total_num_kv_heads=None, bias=False):
        tp_size = dist.get_world_size()
        total_num_kv_heads = total_num_kv_heads or total_num_heads
        self.head_size = head_size
        self.num_heads = divide(total_num_heads, tp_size)
        self.num_kv_heads = divide(total_num_kv_heads, tp_size)
        output_size = (total_num_heads + 2 * total_num_kv_heads) * self.head_size
        super().__init__(hidden_size, output_size, bias)

    def weight_loader(self, param, loaded_weight, loaded_shard_id):
        param_data = param.data
        assert loaded_shard_id in ["q", "k", "v"]
        if loaded_shard_id == "q":
            shard_size = self.num_heads * self.head_size
            shard_offset = 0
        elif loaded_shard_id == "k":
            shard_size = self.num_kv_heads * self.head_size
            shard_offset = self.num_heads * self.head_size
        else:
            shard_size = self.num_kv_heads * self.head_size
            shard_offset = self.num_heads * self.head_size + self.num_kv_heads * self.head_size
        param_data = param_data.narrow(self.tp_dim, shard_offset, shard_size)
        loaded_weight = loaded_weight.chunk(self.tp_size, self.tp_dim)[self.tp_rank]
        param_data.copy_(loaded_weight)
```

### 合并的是 Q、K、V 三个投影

```
完整权重拼接: [W_q (num_heads×head_dim) ; W_k (num_kv_heads×head_dim) ; W_v (num_kv_heads×head_dim)]
```

总输出维度：
```python
output_size = (total_num_heads + 2 * total_num_kv_heads) * self.head_size
```
`total_num_heads` 是 Q，`2 * total_num_kv_heads` 是 K 和 V（GQA 下 kv_heads < heads）。

### GQA 的影响

Qwen3 用 GQA（Grouped Query Attention）：
`num_attention_heads = 16`、`num_key_value_heads = 8`。
所以 Q 部分 16 个头，K/V 各 8 个头。

⚠️ **注意这里和 `Qwen3Attention` 里的切分方式不同**：
- `QKVParallelLinear` 里按**整体维度**切（`super().__init__` 用 complete output_size 除以 tp）
- `Qwen3Attention.__init__` 里 `self.num_heads = total // tp_size`、`self.num_kv_heads = total_kv // tp_size`
  （按 head 数切，用于 reshape）

两者是自洽的：输出维度 = `(num_heads + 2*num_kv_heads) * head_dim`，
按维度切和按 head 切是一致的（因为每个 head 占 head_dim 个连续维度）。

### `weight_loader` 的偏移

本地布局：`[q_shard | k_shard | v_shard]`

| shard_id | offset（本地） | size（本地） |
|----------|--------------|-------------|
| "q" | 0 | `num_heads × head_dim` |
| "k" | `num_heads × head_dim` | `num_kv_heads × head_dim` |
| "v" | `(num_heads + num_kv_heads) × head_dim` | `num_kv_heads × head_dim` |

## 7.7 `VocabParallelEmbedding` 和 `ParallelLMHead`

`nanovllm/layers/embed_head.py:9`：

```python
class VocabParallelEmbedding(nn.Module):

    def __init__(self, num_embeddings, embedding_dim):
        super().__init__()
        self.tp_rank = dist.get_rank()
        self.tp_size = dist.get_world_size()
        assert num_embeddings % self.tp_size == 0
        self.num_embeddings = num_embeddings
        self.num_embeddings_per_partition = self.num_embeddings // self.tp_size
        self.vocab_start_idx = self.num_embeddings_per_partition * self.tp_rank
        self.vocab_end_idx = self.vocab_start_idx + self.num_embeddings_per_partition
        self.weight = nn.Parameter(torch.empty(self.num_embeddings_per_partition, embedding_dim))
        self.weight.weight_loader = self.weight_loader

    def forward(self, x):
        if self.tp_size > 1:
            mask = (x >= self.vocab_start_idx) & (x < self.vocab_end_idx)
            x = mask * (x - self.vocab_start_idx)
        y = F.embedding(x, self.weight)
        if self.tp_size > 1:
            y = mask.unsqueeze(1) * y
            dist.all_reduce(y)
        return y
```

### 词表切分

词表按 rank 切分：每张卡持有 `vocab_size / tp_size` 行 embedding。

```
完整词表: [token0, token1, ..., token_V-1]   (V 行)
rank0:   [token0 .. token_{V/2-1}]
rank1:   [token_{V/2} .. token_{V-1}]
```

### `forward` 的 mask 技巧

问题：输入的 token id 是**全局 id**（0 到 V-1），
但每张卡只有部分 embedding。怎么查？

**(1) 标记超出本地范围的 token**

```python
mask = (x >= self.vocab_start_idx) & (x < self.vocab_end_idx)
```
`mask` 为 True 表示"这个 token 在我这卡里"。

**(2) 索引平移**

```python
x = mask * (x - self.vocab_start_idx)
```
- 属于本卡的：`x - start` 变成**本地索引**
- 不属于的：`mask=0` → 索引变 0（随便查一行，反正不要）

⚠️ `mask * (...)`：`mask` 是 bool，乘整数会转成 0/1，
所以不在范围内的索引全变 0（安全，不会越界）。

**(3) 查表**

```python
y = F.embedding(x, self.weight)
```
每张卡查自己的部分。不属于本卡的行查出来是错的（查到了 token0 的 embedding）。

**(4) 用 mask 抹掉错误结果**

```python
y = mask.unsqueeze(1) * y
```
不属于本卡的 token，其结果被清零。
`unsqueeze(1)` 把 `[N]` 的 mask 变成 `[N, 1]`，广播到 `[N, embedding_dim]`。

**(5) 求和**

```python
dist.all_reduce(y)
```
每张卡的结果求和。对某个 token，只有**持有它的那张卡**结果非零，
其他都是零。所以求和 = 精确取到那一行的 embedding。

> 💡 **这是 TP 里很优雅的一个技巧**：用 mask + all_reduce 做"分布式查表"。
> 也可以用 all_gather 但那样通信量更大。

### `ParallelLMHead`：输出投影

`embed_head.py:45`：

```python
class ParallelLMHead(VocabParallelEmbedding):

    def __init__(self, num_embeddings, embedding_dim, bias=False):
        assert not bias
        super().__init__(num_embeddings, embedding_dim)

    def forward(self, x):
        context = get_context()
        if context.is_prefill:
            last_indices = context.cu_seqlens_q[1:] - 1
            x = x[last_indices].contiguous()
        logits = F.linear(x, self.weight)
        if self.tp_size > 1:
            all_logits = [torch.empty_like(logits) for _ in range(self.tp_size)] if self.tp_rank == 0 else None
            dist.gather(logits, all_logits, 0)
            logits = torch.cat(all_logits, -1) if self.tp_rank == 0 else None
        return logits
```

继承 `VocabParallelEmbedding`，权重形状相同（词表 × hidden）。

### prefill 时只取每个序列的最后一个 token

```python
if context.is_prefill:
    last_indices = context.cu_seqlens_q[1:] - 1
    x = x[last_indices].contiguous()
```

**为什么？** prefill 时一个序列有 N 个 token 的 hidden states，
但它们都对**同一个位置**做预测（预测 prompt 后的第一个生成 token）。
只有最后一个 token 的 hidden state 有用（因果注意力下，
只有最后位置的输出包含了全部上下文）。

`cu_seqlens_q[1:] - 1`：
```
cu_seqlens_q = [0, 5, 8, 12]   (3 个序列长度 5,3,4)
cu_seqlens_q[1:] = [5, 8, 12]
减 1 → [4, 7, 11]  ← 每个序列最后一个 token 的索引 ✓
```

`contiguous()`：高级索引后张量可能不连续，后续 `F.linear` 需要连续内存。

⚠️ **decode 时不需要这步**，因为每个序列本来就只有 1 个 token。

### 词表拼接用 gather 而非 all_reduce

```python
all_logits = [torch.empty_like(logits) for _ in range(self.tp_size)] if self.tp_rank == 0 else None
dist.gather(logits, all_logits, 0)
logits = torch.cat(all_logits, -1) if self.tp_rank == 0 else None
```

这里不能像 embedding 那样 all_reduce（求和）！因为每张卡的 logits
是词表的**不同部分**，要**拼接**成完整词表：

```
rank0: logits[:, 0:V/2]
rank1: logits[:, V/2:V]
→ cat → logits[:, 0:V]   (完整)
```

用 `dist.gather` 汇集到 rank 0。之后只有 rank 0 能采样（第 5 章）。

## 7.8 Qwen3 模型结构

`nanovllm/models/qwen3.py`。

### `Qwen3Attention`

`qwen3.py:14`：

```python
class Qwen3Attention(nn.Module):

    def __init__(self, hidden_size, num_heads, num_kv_heads, max_position=4096*32,
                 head_dim=None, rms_norm_eps=1e-06, qkv_bias=False,
                 rope_theta=10000, rope_scaling=None):
        super().__init__()
        tp_size = dist.get_world_size()
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        assert self.total_num_kv_heads % tp_size == 0
        self.num_kv_heads = self.total_num_kv_heads // tp_size
        self.head_dim = head_dim or hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim ** -0.5
        self.qkv_bias = qkv_bias

        self.qkv_proj = QKVParallelLinear(hidden_size, self.head_dim,
                                          self.total_num_heads, self.total_num_kv_heads, bias=qkv_bias)
        self.o_proj = RowParallelLinear(self.total_num_heads * self.head_dim, hidden_size, bias=False)
        if isinstance(rope_scaling, dict):
            rope_theta = rope_scaling.get("rope_theta", rope_theta)
        self.rotary_emb = get_rope(self.head_dim, rotary_dim=self.head_dim,
                                   max_position=max_position, base=rope_theta)
        self.attn = Attention(self.num_heads, self.head_dim, self.scaling, self.num_kv_heads)
        if not self.qkv_bias:
            self.q_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)
            self.k_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)
```

要点：
- **head 数除以 tp_size**：每张卡算一部分 head
- **`o_proj` 用 RowParallelLinear**：因为输出要拼回完整 hidden_size
- **`scaling = head_dim ** -0.5`**：标准 attention 缩放 1/√d
- **Qwen3 专属的 `q_norm` / `k_norm`**：对每个 head 的 q/k 单独做 RMSNorm
  （这是 Qwen3 相比 Qwen2 的新设计）。只在 `qkv_bias` 为 False 时才加
  （Qwen3 默认无 qkv bias）

### `forward`

`qwen3.py:72`：

```python
def forward(self, positions, hidden_states):
    qkv = self.qkv_proj(hidden_states)
    q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
    q = q.view(-1, self.num_heads, self.head_dim)
    k = k.view(-1, self.num_kv_heads, self.head_dim)
    v = v.view(-1, self.num_kv_heads, self.head_dim)
    if not self.qkv_bias:
        q = self.q_norm(q)
        k = self.k_norm(k)
    q, k = self.rotary_emb(positions, q, k)
    o = self.attn(q, k, v)
    output = self.o_proj(o.flatten(1, -1))
    return output
```

流程：
1. **QKV 投影**（一次大矩阵乘）
2. **split** 成 Q、K、V
3. **reshape** 成多头形状
4. **q_norm/k_norm**（Qwen3 特性）
5. **RoPE** 位置编码
6. **attention**（第 6 章）
7. **flatten + o_proj** 拼回 hidden_size

### `Qwen3MLP`

`qwen3.py:91`：

```python
class Qwen3MLP(nn.Module):

    def __init__(self, hidden_size, intermediate_size, hidden_act):
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(hidden_size, [intermediate_size] * 2, bias=False)
        self.down_proj = RowParallelLinear(intermediate_size, hidden_size, bias=False)
        assert hidden_act == "silu"
        self.act_fn = SiluAndMul()

    def forward(self, x):
        gate_up = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x = self.down_proj(x)
        return x
```

- `gate_up_proj`：**列并行**（合并 gate+up）
- `down_proj`：**行并行**（输入被切，输出需 all_reduce）
- `SiluAndMul`：SwiGLU 激活（见 7.9）

### `Qwen3DecoderLayer`：融合残差

`qwen3.py:120`：

```python
class Qwen3DecoderLayer(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.self_attn = Qwen3Attention(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            max_position=config.max_position_embeddings,
            rms_norm_eps=config.rms_norm_eps,
            qkv_bias=getattr(config, 'attention_bias', True),
            head_dim=getattr(config, 'head_dim', None),
            rope_theta=getattr(config, "rope_theta", 1000000),
            rope_scaling=getattr(config, "rope_scaling", None),
        )
        self.mlp = Qwen3MLP(config.hidden_size, config.intermediate_size, config.hidden_act)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, positions, hidden_states, residual):
        if residual is None:
            hidden_states, residual = self.input_layernorm(hidden_states), hidden_states
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(positions, hidden_states)
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual
```

**残差融合**：标准 Transformer 是这样的：
```python
h = h + attn(norm(h))       # 第一次残差
h = h + mlp(norm(h))        # 第二次残差
```

Nano-vLLM 把残差**累加到 norm 里**：
```python
h, residual = norm(h, residual)    # 一起做：h = norm(h + residual)
```

好处：**少一次读写**。残差的加法融合进 RMSNorm kernel，
减少一次显存往返（`layernorm.py` 的 `add_rms_forward`）。

注意 `residual` 的传递：第一个 layer 传 `None`，
后续 layer 接收上一层的 residual，实现延迟残差累加。

### `Qwen3Model` 和 `Qwen3ForCausalLM`

`qwen3.py:162`：

```python
class Qwen3Model(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([Qwen3DecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, input_ids, positions):
        hidden_states = self.embed_tokens(input_ids)
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(positions, hidden_states, residual)
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states
```

`qwen3.py:186`：

```python
class Qwen3ForCausalLM(nn.Module):
    packed_modules_mapping = {
        "q_proj": ("qkv_proj", "q"),
        "k_proj": ("qkv_proj", "k"),
        "v_proj": ("qkv_proj", "v"),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }

    def __init__(self, config):
        super().__init__()
        self.model = Qwen3Model(config)
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)
        if config.tie_word_embeddings:
            self.lm_head.weight.data = self.model.embed_tokens.weight.data

    def forward(self, input_ids, positions):
        return self.model(input_ids, positions)

    def compute_logits(self, hidden_states):
        return self.lm_head(hidden_states)
```

**`packed_modules_mapping`** 是权重加载的关键（见 7.9）。
它告诉 loader：HF 里的 `q_proj` 权重应该放进本地的 `qkv_proj`，
shard_id 是 `"q"`。

## 7.9 权重加载：`utils/loader.py`

`nanovllm/utils/loader.py`：

```python
def default_weight_loader(param, loaded_weight):
    param.data.copy_(loaded_weight)

def load_model(model, path):
    packed_modules_mapping = getattr(model, "packed_modules_mapping", {})
    for file in glob(os.path.join(path, "*.safetensors")):
        with safe_open(file, "pt", "cpu") as f:
            for weight_name in f.keys():
                for k in packed_modules_mapping:
                    if k in weight_name:
                        v, shard_id = packed_modules_mapping[k]
                        param_name = weight_name.replace(k, v)
                        param = model.get_parameter(param_name)
                        weight_loader = getattr(param, "weight_loader")
                        weight_loader(param, f.get_tensor(weight_name), shard_id)
                        break
                else:
                    param = model.get_parameter(weight_name)
                    weight_loader = getattr(param, "weight_loader", default_weight_loader)
                    weight_loader(param, f.get_tensor(weight_name))
```

### 两种加载路径

**(1) 合并权重**（`q_proj`/`k_proj`/`v_proj`/`gate_proj`/`up_proj`）

```python
if k in weight_name:                       # 比如 "q_proj" in "model.layers.0.self_attn.q_proj.weight"
    v, shard_id = packed_modules_mapping[k]  # ("qkv_proj", "q")
    param_name = weight_name.replace(k, v)   # → "...self_attn.qkv_proj.weight"
    param = model.get_parameter(param_name)
    weight_loader = getattr(param, "weight_loader")
    weight_loader(param, f.get_tensor(weight_name), shard_id)  # 带 shard_id
    break
```

- `replace` 把名字映射到本地的合并权重名
- 传 `shard_id` 告诉 `weight_loader` 放进哪个片段

**(2) 普通权重**（其他所有）

```python
else:
    param = model.get_parameter(weight_name)
    weight_loader = getattr(param, "weight_loader", default_weight_loader)
    weight_loader(param, f.get_tensor(weight_name))
```

用参数自身挂的 `weight_loader`（如果是 TP 层），
否则用默认的直接拷贝。

### `for...else` 的用法

```python
for k in packed_modules_mapping:
    if k in weight_name:
        ...
        break
else:
    # 没 break 过（没匹配到）
    ...
```

`for...else` 的 `else` 在**循环正常结束**（没 break）时执行。
这里表示"这个权重名不属于任何合并模块" → 走普通加载。

### `weight_loader` 挂载机制回顾

还记得 `LinearBase.__init__` 里：
```python
self.weight.weight_loader = self.weight_loader
```
把函数挂到 `Parameter` 对象上。`load_model` 用
`getattr(param, "weight_loader", default_weight_loader)` 取出来。
这是一个巧妙的**多态**：每个层的加载逻辑封装在自己身上，
loader 只管调用。

## 7.10 其余算子

### RMSNorm

`nanovllm/layers/layernorm.py:5`：

```python
class RMSNorm(nn.Module):

    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    @torch.compile
    def rms_forward(self, x):
        orig_dtype = x.dtype
        x = x.float()
        var = x.pow(2).mean(dim=-1, keepdim=True)
        x.mul_(torch.rsqrt(var + self.eps))
        x = x.to(orig_dtype).mul_(self.weight)
        return x

    @torch.compile
    def add_rms_forward(self, x, residual):
        orig_dtype = x.dtype
        x = x.float().add_(residual.float())
        residual = x.to(orig_dtype)
        var = x.pow(2).mean(dim=-1, keepdim=True)
        x.mul_(torch.rsqrt(var + self.eps))
        x = x.to(orig_dtype).mul_(self.weight)
        return x, residual
```

- **`rms_forward`**：标准 RMSNorm。升到 float 算（防溢出），再转回原 dtype
- **`add_rms_forward`**：融合残差。先 `x + residual`，同时把和作为新 residual 返回
- `@torch.compile`：让 torch 编译成融合 kernel，减少显存往返

> 💡 RMSNorm 的公式：`x / sqrt(mean(x²) + eps) * weight`。
> 相比 LayerNorm，不减均值、无 bias。

### SiluAndMul（SwiGLU）

`nanovllm/layers/activation.py:6`：

```python
class SiluAndMul(nn.Module):

    @torch.compile
    def forward(self, x):
        x, y = x.chunk(2, -1)
        return F.silu(x) * y
```

SWiGLU：`silu(gate) * up`。输入是 gate 和 up 拼接的，
chunk 成两半，一半过 silu，再相乘。

### RoPE

`nanovllm/layers/rotary_embedding.py`：

```python
def apply_rotary_emb(x, cos, sin):
    x1, x2 = torch.chunk(x.float(), 2, dim=-1)
    y1 = x1 * cos - x2 * sin
    y2 = x2 * cos + x1 * sin
    return torch.cat((y1, y2), dim=-1).to(x.dtype)

class RotaryEmbedding(nn.Module):

    def __init__(self, head_size, rotary_dim, max_position_embeddings, base):
        super().__init__()
        self.head_size = head_size
        assert rotary_dim == head_size
        inv_freq = 1.0 / (base**(torch.arange(0, rotary_dim, 2, dtype=torch.float) / rotary_dim))
        t = torch.arange(max_position_embeddings, dtype=torch.float)
        freqs = torch.einsum("i,j -> ij", t, inv_freq)
        cos = freqs.cos()
        sin = freqs.sin()
        cache = torch.cat((cos, sin), dim=-1).unsqueeze_(1)
        self.register_buffer("cos_sin_cache", cache, persistent=False)

    @torch.compile
    def forward(self, positions, query, key):
        cos_sin = self.cos_sin_cache[positions]
        cos, sin = cos_sin.chunk(2, dim=-1)
        query = apply_rotary_emb(query, cos, sin)
        key = apply_rotary_emb(key, cos, sin)
        return query, key
```

要点：
- **预计算 cos/sin 缓存**：`cos_sin_cache` 形状 `[max_pos, 1, rotary_dim]`。
  RoPE 的 cos/sin 只和位置有关，可以预先算好，运行时直接查表
- **`inv_freq`**：`1/base^(2i/d)`，即不同维度的旋转频率
- **`register_buffer(..., persistent=False)`**：不作为 `state_dict` 的一部分
  （因为是从参数算出来的，不用保存、不用加载）
- **`@lru_cache(1)` 的 `get_rope`**：保证同配置只建一个实例（省显存）

### Sampler

`nanovllm/layers/sampler.py:5`：

```python
class Sampler(nn.Module):

    @torch.compile
    def forward(self, logits, temperatures):
        logits = logits.float().div_(temperatures.unsqueeze(dim=1))
        probs = torch.softmax(logits, dim=-1)
        sample_tokens = probs.div_(torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)).argmax(dim=-1)
        return sample_tokens
```

逐行：
1. `logits.div_(temperatures.unsqueeze(1))`：温度缩放（`[bs, vocab] / [bs,1]`）
2. `softmax`：转成概率
3. `probs.div_(exponential(1))`：**Gumbel-max 技巧**
4. `argmax`：取最大值的位置

**Gumbel-max 技巧的原理**：
指数分布 `E ~ Exp(1)` 与 Gumbel 分布有等价关系。
`argmax(p_i / e_i)` 等价于从分布 `p` 中采样。
这避免了 `torch.multinomial` 的开销（无法编译 + 逐行处理慢），
且完全可以 `@torch.compile` 融合。

`clamp_min_(1e-10)` 防止除零。

## 7.11 张量并行通信全景

一张表看清每次前向的通信：

| 位置 | 通信 | 何时发生 |
|------|------|---------|
| `VocabParallelEmbedding` | `all_reduce` | 每次前向开头 |
| `QKVParallelLinear` | 无 | —— |
| `Attention` | 无 | —— |
| `o_proj` (RowParallel) | `all_reduce` | 每层 attention 后 |
| `gate_up_proj` (MergedColumn) | 无 | —— |
| `down_proj` (RowParallel) | `all_reduce` | 每层 MLP 后 |
| `ParallelLMHead` | `gather` | 前向末尾 |

**每层 2 次 all_reduce**（o_proj + down_proj），
这是 Tensor Parallel 的固有代价。张量并行靠增大通信换显存，
适合**高带宽互联**（如 NVLink）。

## 7.12 本章小结

- 张量并行的核心：`Linear → 激活 → Linear` 模式，前一个列并行、后一个行并行，
  每个 block 只需一次 `all_reduce`。
- `LinearBase` 把 `weight_loader` 挂到 Parameter 上，实现"加载即切分"的多态。
- `ColumnParallelLinear` 切输出维（无通信）；`RowParallelLinear` 切输入维（需 all_reduce）。
- `MergedColumnParallelLinear` 合并 gate/up；`QKVParallelLinear` 合并 Q/K/V 并按 head 切。
- `VocabParallelEmbedding` 用 mask + all_reduce 做分布式查表；
  `ParallelLMHead` 用 gather 拼接词表，prefill 时只取每序列最后 token。
- Qwen3 特性：q_norm/k_norm、GQA、残差融合进 RMSNorm。
- RoPE 预计算 cos/sin 缓存并用 `@torch.compile` 加速。
- Sampler 用 Gumbel-max 技巧实现可编译的多项分布采样。

**下一章**：[端到端走查](08-walkthrough.md)——把前面所有章节串成一个完整实例。
