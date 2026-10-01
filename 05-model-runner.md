# 第 5 章 ModelRunner：从逻辑序列到 GPU 张量

> 本章目标：理解调度器选出的 `Sequence` 列表，如何被翻译成 GPU 能吃的张量，
> 以及 KV Cache 如何分配、CUDA Graph 如何加速。

## 5.1 ModelRunner 的角色

调度器是纯 CPU 逻辑，它决定"跑谁"。但 GPU 不认识 `Sequence` 对象，
它只认识**张量**。ModelRunner 就是这座桥梁：

```
Scheduler 输出:  [Sequence, Sequence, ...] + is_prefill
                        │
                        ▼  ModelRunner 翻译
GPU 输入:  input_ids, positions, slot_mapping, block_tables, ...
                        │
                        ▼  模型前向
GPU 输出:  logits
                        │
                        ▼  Sampler
token_ids: [int, int, ...]
```

### 每个 GPU 一个 ModelRunner

张量并行时，每张卡跑一个 `ModelRunner` 实例。
rank=0 在主进程，rank>0 在子进程中。

```
进程0 (主)                          进程1
┌────────────────┐                 ┌────────────────┐
│ ModelRunner(0) │                 │ ModelRunner(1) │
│  ├ 模型分片0    │   NCCL all_reduce│  ├ 模型分片1    │
│  ├ KV Cache 0  │◄───────────────►│  ├ KV Cache 1  │
│  └ 采样(只rank0)│   共享内存指令   │  └ 不采样       │
└────────────────┘                 └────────────────┘
        ▲
        │ 主进程直接调用
   LLMEngine
```

⚠️ 注意：**只有 rank=0 采样**。因为 rank>0 不持有完整的输出 logits
（词表是切分的），采样只能由 rank=0 做。

## 5.2 初始化：`ModelRunner.__init__`

`nanovllm/engine/model_runner.py:17`：

```python
def __init__(self, config: Config, rank: int, event: Event | list[Event]):
    self.config = config
    hf_config = config.hf_config
    self.block_size = config.kvcache_block_size
    self.enforce_eager = config.enforce_eager
    self.world_size = config.tensor_parallel_size
    self.rank = rank
    self.event = event

    dist.init_process_group("nccl", "tcp://localhost:2333", world_size=self.world_size, rank=rank)
    torch.cuda.set_device(rank)
    default_dtype = torch.get_default_dtype()
    torch.set_default_dtype(hf_config.dtype)
    torch.set_default_device("cuda")
    self.model = Qwen3ForCausalLM(hf_config)
    load_model(self.model, config.model)
    self.sampler = Sampler()
    self.warmup_model()
    self.allocate_kv_cache()
    if not self.enforce_eager:
        self.capture_cudagraph()
    torch.set_default_device("cpu")
    torch.set_default_dtype(default_dtype)

    if self.world_size > 1:
        if rank == 0:
            self.shm = SharedMemory(name="nanovllm", create=True, size=2**20)
            dist.barrier()
        else:
            dist.barrier()
            self.shm = SharedMemory(name="nanovllm")
            self.loop()
```

### 分步解读

**(1) 初始化 NCCL 进程组**

```python
dist.init_process_group("nccl", "tcp://localhost:2333", world_size=self.world_size, rank=rank)
```

- 后端用 `nccl`（GPU 间高速通信）
- rendezvous 用 `tcp://localhost:2333`（进程发现机制）
- **即使 `world_size=1` 也要初始化**，因为后面的层（linear、embedding）
  会调用 `dist.get_rank()`、`dist.get_world_size()`

> ⚠️ 单卡时这行也会执行，会建立组但通信是空操作。
> 这是为了让 `layers/linear.py` 里的 TP 代码能无条件调用 dist API。

**(2) 设定默认 device 和 dtype**

```python
torch.cuda.set_device(rank)
torch.set_default_dtype(hf_config.dtype)
torch.set_default_device("cuda")
```

设定后，`torch.empty(...)` 之类会**默认在 GPU 上、用模型 dtype**，
省去每处都写 `.cuda()`、`.to(dtype)`。

**(3) 建模型 + 加载权重**

```python
self.model = Qwen3ForCausalLM(hf_config)
load_model(self.model, config.model)
```

`Qwen3ForCausalLM` 在 `torch.set_default_device("cuda")` 生效下构造，
所以权重的 `nn.Parameter` 直接建在 GPU 上。

**(4) 预热 + 分配 KV + 捕获 CUDA Graph**

```python
self.warmup_model()          # 跑一次假数据，触发 CUDA 初始化
self.allocate_kv_cache()     # 测出剩余显存，分配 KV Cache
if not self.enforce_eager:
    self.capture_cudagraph() # 录制各档 batch 的计算图
```

**(5) 恢复默认 device/dtype**

```python
torch.set_default_device("cpu")
torch.set_default_dtype(default_dtype)
```

⚠️ **重要**：初始化完成后把默认设备恢复成 CPU。
否则后面所有张量创建都会莫名跑到 GPU 上。
`prepare_*` 里显式 `.cuda(non_blocking=True)` 是有意为之。

**(6) 多卡：rank>0 进入死循环**

```python
if self.world_size > 1:
    if rank == 0:
        self.shm = SharedMemory(name="nanovllm", create=True, size=2**20)
        dist.barrier()
    else:
        dist.barrier()
        self.shm = SharedMemory(name="nanovllm")
        self.loop()          # ← rank>0 卡在这里，等指令
```

rank>0 的进程构造完基础设施后，**永远停在 `self.loop()`**，
直到收到 `exit` 指令。`self.loop()` 里做什么，见 5.6。

## 5.3 `warmup_model`：为什么要预热

`model_runner.py:91`：

```python
def warmup_model(self):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    max_num_batched_tokens, max_model_len = self.config.max_num_batched_tokens, self.config.max_model_len
    seq_len = min(max_num_batched_tokens, max_model_len)
    num_seqs = min(max_num_batched_tokens // seq_len, self.config.max_num_seqs)
    seqs = [Sequence([0] * seq_len) for _ in range(num_seqs)]
    for seq in seqs:
        seq.num_scheduled_tokens = seq_len
    self.run(seqs, True)
    torch.cuda.empty_cache()
```

### 为什么必须预热？

1. **触发 CUDA 上下文初始化**：第一次跑会有大量一次性开销
2. **让 PyTorch/torch.compile 完成 JIT 编译**：
   `layers/` 里很多 `@torch.compile`，第一次跑要编译，预热后不再卡顿
3. **为 `allocate_kv_cache` 测出峰值显存**：
   不预热的话，`memory_stats()["allocated_bytes.all.peak"]` 不准，
   算出的 block 数会偏大，导致后续 OOM

### 预热的"最坏情况"设计

```python
seq_len = min(max_num_batched_tokens, max_model_len)
num_seqs = min(max_num_batched_tokens // seq_len, self.config.max_num_seqs)
```

构造**最坏情况**的输入：用满 `max_num_batched_tokens` 预算，
序列尽量长。这样预热时的峰值显存 ≥ 实际运行时，
算出的 KV 空间才安全。

例：`max_num_batched_tokens=16384`、`max_model_len=4096`：
```
seq_len = min(16384, 4096) = 4096
num_seqs = min(16384 // 4096, 512) = min(4, 512) = 4
→ 4 条 4096 长的假序列
```

### 假数据的 trick

```python
seqs = [Sequence([0] * seq_len) for _ in range(num_seqs)]
```
全部用 token id 0（没用真实数据）。`Sequence` 只在乎长度。
但注意：假序列的 `block_table` 是空的，所以 `prepare_prefill` 里
```python
if not seq.block_table:    # warmup
    continue
```
会跳过 slot_mapping 的计算（省得为不存在的块算地址）。

## 5.4 `allocate_kv_cache`：按剩余显存反推块数

`model_runner.py:103`，这是全章最精妙的一段：

```python
def allocate_kv_cache(self):
    config = self.config
    hf_config = config.hf_config
    free, total = torch.cuda.mem_get_info()
    used = total - free
    peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
    current = torch.cuda.memory_stats()["allocated_bytes.all.current"]
    num_kv_heads = hf_config.num_key_value_heads // self.world_size
    head_dim = getattr(hf_config, "head_dim", hf_config.hidden_size // hf_config.num_attention_heads)
    block_bytes = 2 * hf_config.num_hidden_layers * self.block_size * num_kv_heads * head_dim * hf_config.dtype.itemsize
    config.num_kvcache_blocks = int(total * config.gpu_memory_utilization - used - peak + current) // block_bytes
    assert config.num_kvcache_blocks > 0
    self.kv_cache = torch.empty(2, hf_config.num_hidden_layers, config.num_kvcache_blocks, self.block_size, num_kv_heads, head_dim)
    layer_id = 0
    for module in self.model.modules():
        if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
            module.k_cache = self.kv_cache[0, layer_id]
            module.v_cache = self.kv_cache[1, layer_id]
            layer_id += 1
```

### (1) 测显存现状

```python
free, total = torch.cuda.mem_get_info()
used = total - free
peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
current = torch.cuda.memory_stats()["allocated_bytes.all.current"]
```

| 变量 | 含义 |
|------|------|
| `total` | GPU 总显存 |
| `free` | 当前空闲 |
| `used` | 已用（含 PyTorch 之外的占用，如 CUDA context）|
| `peak` | PyTorch 分配器历史峰值 |
| `current` | PyTorch 分配器当前分配量 |

### (2) 显存预算公式

```python
config.num_kvcache_blocks = int(total * config.gpu_memory_utilization - used - peak + current) // block_bytes
```

拆解这个公式：

```
可用总预算        = total × 0.9                    (gpu_memory_utilization)
减去当前实际占用   = - used                        (CUDA context 等)
预留峰值余量       = - (peak - current)            ← 见下
                            = - peak + current
════════════════════════════════════════════
可分配给 KV 的字节数 // block_bytes = block 数
```

**为什么要 `- peak + current`？**

`peak - current` 是"PyTorch 分配器曾经多用了多少，现在又还回去了"。
这块空间在**下一次前向时可能又要用**（激活值），所以要预留，不能全给 KV。

- `current` 是当前分配（权重等）
- `peak` 是峰值分配
- `peak - current` = 潜在会再次占用的激活空间

> ⚠️ 这是一个**保守**估算。真实 vLLM 用更精细的 profiling 来测激活峰值。
> Nano-vLLM 用简单公式，在 warmup 后测一次即可，代价是可能浪费一点显存。

### (3) 每块多少字节

```python
num_kv_heads = hf_config.num_key_value_heads // self.world_size
head_dim = getattr(hf_config, "head_dim", hf_config.hidden_size // hf_config.num_attention_heads)
block_bytes = 2 * hf_config.num_hidden_layers * self.block_size * num_kv_heads * head_dim * hf_config.dtype.itemsize
```

逐项：
- **`2`**：K 和 V 各一份
- **`num_hidden_layers`**：每层都要存
- **`block_size`**：一块 256 个 token
- **`num_kv_heads`**：⚠️ 是 `num_key_value_heads`（GQA 的 KV head 数），
  且**除以 world_size**（张量并行把 head 也切了）
- **`head_dim`**：每个 head 的维度
- **`hf_config.dtype.itemsize`**：fp16/bf16 = 2 字节

**Qwen3-0.6B 实例计算**：
```
num_hidden_layers = 28
num_key_value_heads = 8
head_dim = 128
dtype = bf16 → itemsize 2
block_size = 256
world_size = 1

block_bytes = 2 × 28 × 256 × 8 × 128 × 2
            = 2 × 28 × 256 × 2048
            = 29,360,128 字节 ≈ 28 MB per block!
```

28MB 一块，一块存 256 个 token。**所以大模型的 KV Cache 非常吃显存**。

### (4) 一次分配大 tensor

```python
self.kv_cache = torch.empty(2, hf_config.num_hidden_layers, config.num_kvcache_blocks, self.block_size, num_kv_heads, head_dim)
```

形状 `[2, num_layers, num_blocks, block_size, num_kv_heads, head_dim]`：

```
维度0: 2          → K 和 V
维度1: num_layers → 层
维度2: num_blocks → 物理块号  ★ 分页的关键维度
维度3: block_size → 块内位置
维度4: num_kv_heads
维度5: head_dim
```

⚠️ **整个 KV Cache 是一个连续的大 tensor**，从不重新分配。
好处：无碎片、地址可算（见 5.5 的 slot_mapping）。

### (5) 切片挂到每层

```python
layer_id = 0
for module in self.model.modules():
    if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
        module.k_cache = self.kv_cache[0, layer_id]
        module.v_cache = self.kv_cache[1, layer_id]
        layer_id += 1
```

因为 `Attention.__init__` 里设了 `self.k_cache = self.v_cache = torch.tensor([])`，
所以可以用 `hasattr` 找出来。切片是**视图**（view），不复制数据。

遍历顺序恰好是层顺序，所以 `layer_id` 从 0 递增能对上。

### 拓扑图

```
self.kv_cache  (整个大 tensor，约 N GB)
    │
    ├─ [0] → K 缓存 ──┬─ [layer0] → 挂到 layers[0].self_attn.attn.k_cache
    │                 ├─ [layer1] → ...
    │                 └─ ...
    └─ [1] → V 缓存 ──┴─ 同理
```

## 5.5 三种 prepare：组装张量

这是 ModelRunner 最核心的工作。根据 prefill/decode 组装不同的张量。

### 5.5.1 `prepare_block_tables`

`model_runner.py:123`：

```python
def prepare_block_tables(self, seqs: list[Sequence]):
    max_len = max(len(seq.block_table) for seq in seqs)
    block_tables = [seq.block_table + [-1] * (max_len - len(seq.block_table)) for seq in seqs]
    block_tables = torch.tensor(block_tables, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
    return block_tables
```

把变长的块表**对齐成矩阵**（padding 用 -1）：

```
seq A: block_table = [3, 7]        → [3, 7, -1, -1]
seq B: block_table = [0, 1, 5, 9]  → [0, 1, 5, 9]
max_len = 4
结果矩阵: [[3,7,-1,-1],
          [0,1,5,9]]
```

- `-1` 是 padding 标记，attention 会忽略
- `pin_memory=True`：贴在固定内存，`non_blocking=True` 的 H2D 拷贝可异步，
  与 CUDA Graph 配合（见 5.7）

### 5.5.2 `prepare_prefill`

`model_runner.py:129`，最复杂的一个：

```python
def prepare_prefill(self, seqs: list[Sequence]):
    input_ids = []
    positions = []
    cu_seqlens_q = [0]
    cu_seqlens_k = [0]
    max_seqlen_q = 0
    max_seqlen_k = 0
    slot_mapping = []
    block_tables = None
    for seq in seqs:
        start = seq.num_cached_tokens
        seqlen_q = seq.num_scheduled_tokens
        end = start + seqlen_q
        seqlen_k = end
        input_ids.extend(seq[start:end])
        positions.extend(range(start, end))
        cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
        cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)
        max_seqlen_q = max(seqlen_q, max_seqlen_q)
        max_seqlen_k = max(seqlen_k, max_seqlen_k)
        if not seq.block_table:    # warmup
            continue
        start_block = start // self.block_size
        end_block = (end + self.block_size - 1) // self.block_size
        for i in range(start_block, end_block):
            slot_start = seq.block_table[i] * self.block_size
            if i == start_block:
                slot_start += start % self.block_size
            if i != end_block - 1:
                slot_end = seq.block_table[i] * self.block_size + self.block_size
            else:
                slot_end = seq.block_table[i] * self.block_size + end - i * self.block_size
            slot_mapping.extend(range(slot_start, slot_end))
    if cu_seqlens_k[-1] > cu_seqlens_q[-1]:    # prefix cache
        block_tables = self.prepare_block_tables(seqs)
    input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
    positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
    cu_seqlens_q = torch.tensor(cu_seqlens_q, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
    cu_seqlens_k = torch.tensor(cu_seqlens_k, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
    slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
    set_context(True, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, None, block_tables)
    return input_ids, positions
```

#### 关键变量解释

对每个序列：

```python
start = seq.num_cached_tokens      # 从哪个 token 开始算
seqlen_q = seq.num_scheduled_tokens # 要算多少 query
end = start + seqlen_q             # 算到哪
seqlen_k = end                     # key 长度 = 前面全部 + 本次
```

**为什么 seqlen_k = end？** prefill 是因果注意力，
第 i 个 query 能看到 0..i 的所有 key。所以 key 的范围是 0..end。

#### `cu_seqlens`：变长序列的"分段累计长度"

这是 flash-attn varlen 的核心参数。假设 3 个序列，
query 长度分别是 5、3、4：

```python
cu_seqlens_q = [0, 5, 8, 12]    # 累计和！
```

含义：第 0 个序列的 q 在 `input_ids[0:5]`，第 1 个在 `[5:8]`，第 2 个在 `[8:12]`。

**为什么要这个？** flash-attn 要把多个变长序列拼成一个扁平的一维张量
（避免 padding 浪费），就需要知道每个序列的边界。`cu_seqlens`
（cumulative sequence lengths）就是边界数组。

```
扁平 input_ids: [s0t0 s0t1 s0t2 s0t3 s0t4 | s1t0 s1t1 s1t2 | s2t0 s2t1 s2t2 s2t3]
                   ↑ 序列0 (5个)             ↑ 序列1 (3个)      ↑ 序列2 (4个)
cu_seqlens_q = [0, 5, 8, 12]
```

#### `cu_seqlens_k` 与 `cu_seqlens_q` 的区别

prefill 时，如果**有前缀缓存**，`seqlen_k > seqlen_q`：

```
序列 A: prompt 600 token，命中 512 前缀缓存
  start = 512（跳过已缓存的）  seqlen_q = 88（只算新的）
  seqlen_k = 600（但 key 要看全部 600 个）
```

因为注意力需要访问**所有**历史 key（含缓存的），但只需要计算 88 个新 query。
所以 `cu_seqlens_k` ≠ `cu_seqlens_q`。这也是为什么后面要判断：

```python
if cu_seqlens_k[-1] > cu_seqlens_q[-1]:    # prefix cache
    block_tables = self.prepare_block_tables(seqs)
```

只有存在前缀缓存（k 比 q 长）时，才需要按块表读历史 KV。

#### `slot_mapping`：KV 该写到哪个物理位置

这是**分页写入**的核心。`slot_mapping[i]` 告诉你：
扁平序列的第 i 个 token 的 K/V 应该写到 KV Cache 的**哪个槽位**。

槽位 = `block_id × block_size + 块内偏移`。

```python
start_block = start // self.block_size
end_block = (end + self.block_size - 1) // self.block_size
for i in range(start_block, end_block):
    slot_start = seq.block_table[i] * self.block_size
    if i == start_block:
        slot_start += start % self.block_size
    if i != end_block - 1:
        slot_end = seq.block_table[i] * self.block_size + self.block_size
    else:
        slot_end = seq.block_table[i] * self.block_size + end - i * self.block_size
    slot_mapping.extend(range(slot_start, slot_end))
```

**逐行解释**：
- `start_block`：起始 token 属于哪个块
- `end_block`：结束 token 属于哪个块（向上取整）
- 对涉及的每个块：
  - `slot_start`：块的物理起点。如果是起始块，还要加上块内偏移 `start % block_size`
  - `slot_end`：块的物理终点。如果是最后一块，用 `end` 算出真实终点
    （块可能没满）；否则整块
  - `extend(range(...))`：把这一段的槽位号逐个加进去

**具体例子**：block_size=256，序列 A（prompt=600，命中 512）。
`start=512, end=600`。
- `start_block = 512//256 = 2`，`end_block = (600+255)//256 = 3`
- 只有块2：
  - `i=2` 是 `start_block` → `slot_start = block_table[2]*256 + 512%256 = block_table[2]*256 + 0`
  - `i=2` 也是 `end_block-1` → `slot_end = block_table[2]*256 + 600 - 2*256 = block_table[2]*256 + 88`
  - 槽位 = `[block_table[2]*256 + 0 .. block_table[2]*256 + 87]`，共 88 个 ✓

**为什么这么绕？** 因为要处理三种情况：
1. 起始块可能从中间开始（前缀缓存命中）
2. 最后的块可能没填满
3. 中间的块是完整的

#### `positions`

```python
positions.extend(range(start, end))
```
位置编码的索引。前缀缓存命中时从 512 开始，而不是 0——
这样 RoPE 的位置信息才对。

### 5.5.3 `prepare_decode`

`model_runner.py:172`，比 prefill 简单得多：

```python
def prepare_decode(self, seqs: list[Sequence]):
    input_ids = []
    positions = []
    slot_mapping = []
    context_lens = []
    for seq in seqs:
        input_ids.append(seq.last_token)
        positions.append(len(seq) - 1)
        context_lens.append(len(seq))
        slot_mapping.append(seq.block_table[-1] * self.block_size + seq.last_block_num_tokens - 1)
    input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
    positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
    slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
    context_lens = torch.tensor(context_lens, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
    block_tables = self.prepare_block_tables(seqs)
    set_context(False, slot_mapping=slot_mapping, context_lens=context_lens, block_tables=block_tables)
    return input_ids, positions
```

每个序列只贡献 **1 个 token**：

| 张量 | 内容 | 说明 |
|------|------|------|
| `input_ids` | `seq.last_token` | 只喂最后一个 token |
| `positions` | `len(seq) - 1` | 该 token 的位置 |
| `context_lens` | `len(seq)` | 整个上下文长度（供 flash-attn 定位） |
| `slot_mapping` | `block_table[-1]*256 + last_block_num_tokens - 1` | 新 KV 写入最后一块的下一个槽位 |
| `block_tables` | 同 prefill | 用于读取历史 KV |

#### `slot_mapping` 的推导

新 token 的 KV 应该写在当前最后一块的**下一个空位**：

```
最后一块的物理起点 = block_table[-1] * block_size
块内偏移 = last_block_num_tokens - 1
```

例：序列有 600 个 token，最后一块有 88 个（`last_block_num_tokens=88`）。
新 token 是第 601 个，写入偏移 88（0-indexed），即块内第 89 个位置。

⚠️ 注意 `len(seq)` 此时是 **601**（prompt 600 + 已生成 1 个），
但 `last_block_num_tokens` 基于 `num_tokens` 算，要小心对齐。
实际上 `may_append` 已经在需要时扩了块，所以这里算的槽位一定合法。

#### 为什么 decode 不需要 `cu_seqlens`？

因为每个序列都是等长（都是 1 个 query）。没有变长问题，
flash-attn 的 `flash_attn_with_kvcache` 用 `context_lens` 就能处理。

### 5.5.4 `prepare_sample`

`model_runner.py:190`：

```python
def prepare_sample(self, seqs: list[Sequence]):
    temperatures = [seq.temperature for seq in seqs]
    temperatures = torch.tensor(temperatures, dtype=torch.float32, pin_memory=True).cuda(non_blocking=True)
    return temperatures
```

把每个序列的温度收集成张量。因为采样是逐序列的，
不同序列可以有不同温度。

### 5.5.5 全局 Context：跨层的隐式传参

上面三个 prepare 都用 `set_context(...)` 把信息塞进一个**全局单例**：

`nanovllm/utils/context.py`：

```python
@dataclass(slots=True)
class Context:
    is_prefill: bool = False
    cu_seqlens_q: torch.Tensor | None = None
    cu_seqlens_k: torch.Tensor | None = None
    max_seqlen_q: int = 0
    max_seqlen_k: int = 0
    slot_mapping: torch.Tensor | None = None
    context_lens: torch.Tensor | None = None
    block_tables: torch.Tensor | None = None

_CONTEXT = Context()

def get_context():
    return _CONTEXT

def set_context(is_prefill, cu_seqlens_q=None, ...):
    global _CONTEXT
    _CONTEXT = Context(is_prefill, cu_seqlens_q, ...)

def reset_context():
    global _CONTEXT
    _CONTEXT = Context()
```

**为什么用全局单例，而不是参数传递？**

因为在模型前向的**深处**（每层的 Attention）都需要这些信息，
但 model 的 `forward(input_ids, positions)` 签名不想塞这么多参数
（否则每个子模块的 forward 都要改签名）。

所以用全局单例做一个"侧信道"。第 6 章的 `Attention.forward` 里
`context = get_context()` 就是这么拿到的。

⚠️ 代价：**不可重入**。多个请求并发跑在同一个 ModelRunner 上时，
这个全局状态是共享的。但因为在 Nano-vLLM 里，
一次 `run` 是一个完整的原子操作（单线程、顺序），所以没问题。

## 5.6 多卡通信：共享内存 + Event

`model_runner.py:61-89`：

```python
def loop(self):
    while True:
        method_name, args = self.read_shm()
        self.call(method_name, *args)
        if method_name == "exit":
            break

def read_shm(self):
    assert self.world_size > 1 and self.rank > 0
    self.event.wait()
    n = int.from_bytes(self.shm.buf[0:4], "little")
    method_name, *args = pickle.loads(self.shm.buf[4:n+4])
    self.event.clear()
    return method_name, args

def write_shm(self, method_name, *args):
    assert self.world_size > 1 and self.rank == 0
    data = pickle.dumps([method_name, *args])
    n = len(data)
    self.shm.buf[0:4] = n.to_bytes(4, "little")
    self.shm.buf[4:n+4] = data
    for event in self.event:
        event.set()

def call(self, method_name, *args):
    if self.world_size > 1 and self.rank == 0:
        self.write_shm(method_name, *args)
    method = getattr(self, method_name, None)
    return method(*args)
```

### 通信协议

rank=0 想把 `run(seqs, is_prefill)` 这个调用发给其他 rank：

**rank=0 侧（write_shm）**：
1. `pickle.dumps([method_name, *args])` 序列化
2. 前 4 字节写长度（小端）
3. 后面写数据
4. 对所有 event 调 `set()` 唤醒子进程

**rank>0 侧（read_shm）**：
1. `self.event.wait()` 阻塞等信号
2. 读前 4 字节得到长度
3. 反序列化
4. `event.clear()` 复位，等待下次

### 为什么用共享内存而不是 pickle.send？

因为 `Sequence` 对象里含 `block_table`，且数据量可能不小。
共享内存是**零拷贝**的（`self.shm.buf` 直接是内存视图），
比走管道/socket 快。1MB 的 buffer（`2**20`）足够装下一个 step 的指令。

⚠️ **`call()` 在 rank=0 上同时"广播 + 本地执行"**：
先 write_shm 通知别人，然后自己也 `method(*args)`。
所以多卡时 rank=0 也参与计算。

⚠️ **NCCL 通信不在写共享内存里**。ModelRunner 里的所有
`dist.all_reduce`、`dist.gather` 都是各 rank 独立执行，
靠 NCCL 同步。共享内存只传"这一步跑什么"的指令。

### 为什么 rank>0 不采样？

看 `run()`：

```python
def run(self, seqs: list[Sequence], is_prefill: bool) -> list[int]:
    input_ids, positions = self.prepare_prefill(seqs) if is_prefill else self.prepare_decode(seqs)
    temperatures = self.prepare_sample(seqs) if self.rank == 0 else None
    logits = self.run_model(input_ids, positions, is_prefill)
    token_ids = self.sampler(logits, temperatures).tolist() if self.rank == 0 else None
    reset_context()
    return token_ids
```

`rank != 0` 时不准备温度、不采样（因为没完整 logits），
返回 `None`。只有 rank=0 返回 token。

## 5.7 CUDA Graph：把小 batch decode 的调度开销抹掉

### 为什么需要 CUDA Graph？

GPU 执行一个 kernel 时，CPU 要做一系列"启动"动作：
分配内存、算 grid 大小、下发命令……这些**启动开销**对单个 kernel 可能
只要几微秒，但 decode 时每步有上百个 kernel，累积起来就显著了。

尤其小 batch（比如只跑 1、2 个序列）时，GPU 计算量很小，
CPU 启动开销反而成了瓶颈——GPU 大部分时间在**等 CPU 下命令**。

**CUDA Graph** 的解法：把一整个 batch 的所有 kernel 启动
**录制**成一张图，之后直接 `graph.replay()`，GPU 一连串执行，
CPU 只发一条命令。

### 录制：`capture_cudagraph`

`model_runner.py:223`：

```python
@torch.inference_mode()
def capture_cudagraph(self):
    config = self.config
    hf_config = config.hf_config
    max_bs = min(self.config.max_num_seqs, 512)
    max_num_blocks = (config.max_model_len + self.block_size - 1) // self.block_size
    input_ids = torch.zeros(max_bs, dtype=torch.int64)
    positions = torch.zeros(max_bs, dtype=torch.int64)
    slot_mapping = torch.zeros(max_bs, dtype=torch.int32)
    context_lens = torch.zeros(max_bs, dtype=torch.int32)
    block_tables = torch.zeros(max_bs, max_num_blocks, dtype=torch.int32)
    outputs = torch.zeros(max_bs, hf_config.hidden_size)
    self.graph_bs = [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))
    self.graphs = {}
    self.graph_pool = None

    for bs in reversed(self.graph_bs):
        graph = torch.cuda.CUDAGraph()
        set_context(False, slot_mapping=slot_mapping[:bs], context_lens=context_lens[:bs], block_tables=block_tables[:bs])
        outputs[:bs] = self.model(input_ids[:bs], positions[:bs])    # warmup
        with torch.cuda.graph(graph, self.graph_pool):
            outputs[:bs] = self.model(input_ids[:bs], positions[:bs])    # capture
        if self.graph_pool is None:
            self.graph_pool = graph.pool()
        self.graphs[bs] = graph
        torch.cuda.synchronize()
        reset_context()

    self.graph_vars = dict(
        input_ids=input_ids,
        positions=positions,
        slot_mapping=slot_mapping,
        context_lens=context_lens,
        block_tables=block_tables,
        outputs=outputs,
    )
```

### 关键设计点

**(1) 预分配一批固定大小的输入张量**

```python
input_ids = torch.zeros(max_bs, ...)
...
block_tables = torch.zeros(max_bs, max_num_blocks, ...)
```

所有档位的 graph 共享这些**静态张量**。回放时只需要往它们里填数据，
不用重新分配（CUDA Graph 要求内存地址固定）。

**(2) 各档 batch size**

```python
self.graph_bs = [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))
```
覆盖 1、2、4、8、16、32、48、...、512。
小 batch 密一些（常见），大 batch 稀一些（省内存）。

**(3) 逆序录制 + 共享内存池**

```python
for bs in reversed(self.graph_bs):
    ...
    with torch.cuda.graph(graph, self.graph_pool):
        ...
    if self.graph_pool is None:
        self.graph_pool = graph.pool()
```

从**最大** bs 开始录，且所有 graph 共享一个 `graph_pool`
（内存池）。逆序是为了：大图先建池，小图复用它。
这样能显著省显存。

**(4) 先 warmup 再 capture**

```python
outputs[:bs] = self.model(...)   # warmup
with torch.cuda.graph(graph, self.graph_pool):
    outputs[:bs] = self.model(...)   # capture
```
CUDA Graph 要求捕获期间没有内存分配等副作用。
先跑一次让 `torch.compile`、cuBLAS 等完成初始化。

### 回放：`run_model`

`model_runner.py:195`：

```python
@torch.inference_mode()
def run_model(self, input_ids: torch.Tensor, positions: torch.Tensor, is_prefill: bool):
    if is_prefill or self.enforce_eager or input_ids.size(0) > 512:
        return self.model.compute_logits(self.model(input_ids, positions))
    else:
        bs = input_ids.size(0)
        context = get_context()
        graph = self.graphs[next(x for x in self.graph_bs if x >= bs)]
        graph_vars = self.graph_vars
        graph_vars["input_ids"][:bs] = input_ids
        graph_vars["positions"][:bs] = positions
        graph_vars["slot_mapping"].fill_(-1)
        graph_vars["slot_mapping"][:bs] = context.slot_mapping
        graph_vars["context_lens"].zero_()
        graph_vars["context_lens"][:bs] = context.context_lens
        graph_vars["block_tables"][:bs, :context.block_tables.size(1)] = context.block_tables
        graph.replay()
        return self.model.compute_logits(graph_vars["outputs"][:bs])
```

**(1) 什么时候走 eager（不用 graph）？**

```python
if is_prefill or self.enforce_eager or input_ids.size(0) > 512:
```
- **prefill**：形状变化太大（变长），不适合固定图
- **enforce_eager**：用户显式禁用
- **bs > 512**：没有对应的图

**(2) 找出"盖得住"的档位**

```python
graph = self.graphs[next(x for x in self.graph_bs if x >= bs)]
```
比如 bs=20，找最小的 ≥20 的档位 → 32。
用 32 的图跑 20 个，多出来的位置填垃圾（反正不看）。

**(3) 填数据到静态张量**

把动态的 `input_ids` 等拷进 `graph_vars` 的对应位置。
注意**清空尾部**（`fill_(-1)`、`zero_()`），避免上一轮的残留数据污染。

**(4) 回放**

```python
graph.replay()
```
一条命令，GPU 执行整张图。输出在 `graph_vars["outputs"]` 里。

**(5) 取结果**

```python
return self.model.compute_logits(graph_vars["outputs"][:bs])
```
只取前 bs 行（有效部分）。注意 `compute_logits`（LM head 投影）
**在图外**执行——因为它是大矩阵乘，放图里意义不大，且输出 vocab 很大。

### 为什么 prefill 不用 CUDA Graph？

因为 prefill 的**形状高度动态**：序列长度、batch 大小每次都变。
CUDA Graph 要求形状固定（张量地址固定），所以 prefill 不适合。
而 decode 每个序列都贡献 1 个 token，只有 batch 大小在变，
可以用"几个档位盖住"的策略。

## 5.8 `run()`：一次完整执行

`model_runner.py:214`：

```python
def run(self, seqs: list[Sequence], is_prefill: bool) -> list[int]:
    input_ids, positions = self.prepare_prefill(seqs) if is_prefill else self.prepare_decode(seqs)
    temperatures = self.prepare_sample(seqs) if self.rank == 0 else None
    logits = self.run_model(input_ids, positions, is_prefill)
    token_ids = self.sampler(logits, temperatures).tolist() if self.rank == 0 else None
    reset_context()
    return token_ids
```

四步：
1. **prepare**：组装张量，写进全局 Context
2. **run_model**：模型前向（可能走 CUDA Graph）
3. **sample**：采样出 token（只 rank=0）
4. **reset_context**：清空全局状态，准备下一轮

⚠️ `reset_context()` 很重要。如果不重置，下一轮的 prepare 若忘记覆盖
某个字段，会残留上一轮的值造成 bug。

## 5.9 本章小结

- ModelRunner 是"逻辑序列 → GPU 张量"的翻译器，每 GPU 一个。
- 初始化流程：NCCL → 建模型 → 加载权重 → warmup → 分配 KV → 捕获 CUDA Graph。
- **KV Cache 是一个巨大的、永不重分配的张量**，形状
  `[2, layers, blocks, block_size, kv_heads, head_dim]`，切片挂到每层。
- `block_bytes` 公式里的 `num_kv_heads` 要除以 `world_size`（TP 切了 head）。
- 块数由 `total × utilization - used - peak + current` 反推。
- **prepare_prefill**：变长序列拼接，用 `cu_seqlens` 标记边界，
  `slot_mapping` 指明 KV 写入位置。
- **prepare_decode**：每序列 1 个 token，用 `block_tables` 读历史 KV。
- **全局 Context 单例**做跨层隐式传参（代价是不可重入）。
- **多卡通信**：rank=0 通过共享内存 + Event 广播指令给 rank>0；
  rank>0 停在 `loop()` 等指令，不采样。
- **CUDA Graph**：把 decode 的多 kernel 启动录成图回放，
  用几个 batch 档位 + 静态张量 + 共享内存池；prefill 不用。

**下一章**：[Attention 与 KV Cache](06-attention.md)——KV 到底怎么存、attention 怎么算。
