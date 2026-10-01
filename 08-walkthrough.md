# 第 8 章 端到端走查：三个请求的一生

> 本章目标：把前七章串起来，用一个具体到每个数字的例子，
> 亲眼看着三个请求从创建到输出的完整旅程。

## 8.1 舞台设定

为了让每个数字都能手算，我们用一个**极致简化的配置**：

```
模型:          一个玩具模型
block_size:    4        (真实是 256，这里缩小便于手算)
max_num_batched_tokens: 12
max_num_seqs:  4
max_model_len: 100

KV Cache:      8 个物理块 (块号 0..7)，每块 4 槽位 → 共 32 槽位

请求:
  A: prompt = [a0,a1,a2,a3,a4]    (5 token)
  B: prompt = [b0,b1,b2]          (3 token)
  C: prompt = [a0,a1,a2,a3,a4]    (5 token，和 A 前缀相同！)
```

⚠️ A 和 C 的 prompt 完全相同，这是为了演示**前缀缓存**。

假设 EOS 永远不会被采到，用 max_tokens 控制结束：
- A: max_tokens = 2
- B: max_tokens = 3
- C: max_tokens = 2

## 8.2 初始状态

```
llm.generate([A, B, C], params)
  ↓ add_request × 3
```

三个 `Sequence` 入 `waiting`：

```
waiting = [A, B, C]
running = []

A: seq_id=0, num_tokens=5, num_prompt_tokens=5, num_cached_tokens=0, block_table=[]
B: seq_id=1, num_tokens=3, num_prompt_tokens=3, num_cached_tokens=0, block_table=[]
C: seq_id=2, num_tokens=5, num_prompt_tokens=5, num_cached_tokens=0, block_table=[]

free_block_ids = [0,1,2,3,4,5,6,7]
used_block_ids = {}
hash_to_block_id = {}
```

## 8.3 Step 1：三个 prefill

### schedule()

**阶段一：prefill**

**队首 A**：
- `remaining = 12`
- `block_table` 空 → `can_allocate(A)`：
  - 遍历 A 的 `num_blocks - 1 = (5+3)//4 - 1 = 1 - 1 = 0` 块 → **循环不执行**
  - `num_cached_blocks = 0`, `num_new_blocks = num_blocks = 2`
  - `free(8) >= 2` ✓ → 返回 **0**
- `num_tokens = 5 - 0*4 = 5`
- `remaining(12) >= 5`，可以选
- `allocate(A, 0)`：无前缀命中，分配 2 个新块 → 假设分配块 0、1
  - `A.block_table = [0, 1]`，`A.num_cached_tokens = 0`
- `A.num_scheduled_tokens = min(5, 12) = 5`
- `num_batched_tokens = 5`
- `0 + 5 == 5` ✓ 算完 → `A.status = RUNNING`，移到 running
- `scheduled_seqs = [A]`

**队首 B**：
- `remaining = 12 - 5 = 7`
- `can_allocate(B)`：`num_blocks = (3+3)//4 = 1`，循环 0 次，返回 0
- `num_tokens = 3`
- `remaining(7) >= 3` ✓
- `allocate(B, 0)`：分配 1 个新块 → 块 2
  - `B.block_table = [2]`
- `B.num_scheduled_tokens = 3`，`num_batched_tokens = 8`
- 算完 → RUNNING
- `scheduled_seqs = [A, B]`

**队首 C**：
- `remaining = 12 - 8 = 4`
- `can_allocate(C)`：
  - C 的 `num_blocks - 1 = 1`，循环 i=0 一次
  - `token_ids = C.block(0) = [a0,a1,a2,a3]`
  - `h = compute_hash([a0,a1,a2,a3], -1)`
  - ⚠️ A 已经算完并登记了块！等等——**hash_blocks 是在 postprocess 里做的**，
    现在还没到 postprocess。所以 `hash_to_block_id` 还是空的
  - `hash_to_block_id.get(h, -1) = -1` → break
  - `num_cached_blocks = 0`, `num_new_blocks = 2`
  - `free(8-3=5) >= 2` ✓ → 返回 **0**
- `num_tokens = 5`
- `remaining(4) < 5` **且 `scheduled_seqs` 非空** → **`break`！**（C 是第二个及以后，不允许分块）

C 这步不跑。

**返回 `([A, B], True)`**

```
waiting = [C]
running = [A, B]
```

> 💡 注意这里 C 和 A 前缀相同，但**这一步没能复用**。
> 因为 A 的块 hash 要到 postprocess 才登记。所以前缀缓存
> 只在**后续 step** 才有机会命中。这是很重要的一课。

### run()

`ModelRunner.run([A, B], is_prefill=True)`：

`prepare_prefill([A, B])`：
- A: start=0, end=5, seqlen_q=5, seqlen_k=5
  - `slot_mapping`: 
    - start_block=0, end_block=(5+3)//4=2
    - 块0(block_table[0]=0): i==start → slot_start=0*4+0=0；i!=end-1 → slot_end=0*4+4=4 → `[0,1,2,3]`
    - 块1(block_table[1]=1): i!=start → slot_start=1*4=4；i==end-1 → slot_end=1*4+5-1*4=1 → `[4]`... 
    
    等等，`slot_end = 1*4 + 5 - 1*4 = 4 + 1 = 5`。修正：`end - i*block_size = 5 - 1*4 = 1`，
    所以 `slot_end = block_table[1]*4 + 1 = 4 + 1 = 5`。`range(4, 5) = [4]` ✓
  - `slot_mapping` 累计 = `[0,1,2,3,4]`
- B: start=0, end=3, seqlen_q=3, seqlen_k=3
  - start_block=0, end_block=(3+3)//4=1
  - 块2(block_table[0]=2): slot_start=2*4=8, slot_end=2*4+3=11 → `[8,9,10]`
  - `slot_mapping` 累计 = `[0,1,2,3,4, 8,9,10]`

- `cu_seqlens_q = [0, 5, 8]`
- `cu_seqlens_k = [0, 5, 8]`（无前缀缓存，k==q）
- `cu_seqlens_k[-1] (8) > cu_seqlens_q[-1] (8)`? **否** → `block_tables = None`
- `max_seqlen_q = max_seqlen_k = 5`

平铺输入：
```
input_ids = [a0,a1,a2,a3,a4, b0,b1,b2]
positions = [0,1,2,3,4, 0,1,2]
```

模型前向 → KV 写入：
```
块0 (槽0-3):  A[a0..a3]
块1 (槽4):    A[a4]
块2 (槽8-10): B[b0..b2]
```

`ParallelLMHead`：prefill 时取 `cu_seqlens_q[1:]-1 = [5, 8] - 1 = [4, 7]`
→ 取平铺 hidden 的第 4、7 个（A 的 a4、B 的 b2）算 logits。

采样：
- A → `t5`（第 6 个 token）
- B → `t3`（第 4 个 token）

### postprocess()

**对 A**：
- `hash_blocks(A)`：`start = 0//4 = 0`, `end = 0+5 // 4 = 1`
  - 登记块0：`h0 = hash([a0,a1,a2,a3], -1)`，`Block[0].update(h0, [a0..a3])`
  - `hash_to_block_id[h0] = 0`
  - ⚠️ 块1没满（只有 1 个 token），不登记
- `A.num_cached_tokens += 5` → 5
- `is_prefill(True) and 5 < 6`? `num_tokens` 还是 5（还没 append）→ `5 < 5` False → 继续
- `append_token(t5)` → `A.num_tokens = 6`, `last_token = t5`
- `num_completion_tokens = 6 - 5 = 1 != 2` → 继续

**对 B**：
- `hash_blocks(B)`：`start = 0`, `end = 3//4 = 0` → `start == end` → **直接 return**（块没满）
- `B.num_cached_tokens = 3`
- `append_token(t3)` → `B.num_tokens = 4`
- `num_completion_tokens = 1 != 3` → 继续

### Step 1 结束状态

```
waiting = [C]
running = [A, B]

A: num_tokens=6, num_cached_tokens=5, block_table=[0,1], last_token=t5
B: num_tokens=4, num_cached_tokens=3, block_table=[2], last_token=t3
C: num_tokens=5, num_cached_tokens=0, block_table=[]

free = [3,4,5,6,7]
used = {0,1,2}
hash_to_block_id = {h0: 0}
```

## 8.4 Step 2：C 的前缀缓存命中 + A、B 的 decode

### schedule()

**阶段一：prefill（waiting=[C]）**

**队首 C**：
- `block_table` 空 → `can_allocate(C)`：
  - C 的 `num_blocks = 2`，循环 `range(1)` → i=0
  - `token_ids = C.block(0) = [a0,a1,a2,a3]`
  - `h = compute_hash([a0,a1,a2,a3], -1) = h0` ✓ 和 A 登记的一样！
  - `hash_to_block_id.get(h0) = 0`，不是 -1
  - `Block[0].token_ids == [a0..a3]` ✓ 匹配
  - `num_cached_blocks = 1`
  - 块0 在 `used` 里（A 在用）→ `num_new_blocks = 2 - 1 = 1`
  - `free(5) >= 1` ✓ → 返回 **1**
- `num_tokens = 5 - 1*4 = 1`（只需算 C 的第 5 个 token！）
- `remaining = 12 >= 1` ✓
- `allocate(C, 1)`：
  - 块0 在 used → `ref_count` 1→2，`C.block_table = [0]`
  - 分配 1 个新块（块3）→ `C.block_table = [0, 3]`
  - `C.num_cached_tokens = 1*4 = 4`
- `C.num_scheduled_tokens = 1`
- `4 + 1 == 5` ✓ 算完 → RUNNING
- `scheduled_seqs = [C]`

**返回 `([C], True)`** —— 只有 C，是 prefill！

⚠️ **注意：这步只跑 prefill，不跑 A、B 的 decode。**
因为有 prefill 任务时优先做 prefill。A、B 这一步被耽搁了。

> 💡 这就是 `schedule` "prefill 优先"策略的直接后果：
> 新请求的 prefill 会**抢占**正在 decode 的请求的时间片。
> 好处是新请求的 TTFT 短，代价是已有请求的 TPOT 略增。

### run()：C 的前缀缓存 prefill

`prepare_prefill([C])`：
- `start = C.num_cached_tokens = 4`, `end = 4 + 1 = 5`
- `seqlen_q = 1`, `seqlen_k = 5`
- `input_ids = [a4]`（只算第 5 个 token）
- `positions = [4]`
- `cu_seqlens_q = [0, 1]`
- `cu_seqlens_k = [0, 5]`
- `cu_seqlens_k[-1] (5) > cu_seqlens_q[-1] (1)` ✓ → **需要 block_tables！**
  - `block_tables = [[0, 3]]`
- `slot_mapping`：
  - start_block = 4//4 = 1, end_block = (5+3)//4 = 2
  - 块1(block_table[1]=3): i==start → slot_start=3*4+4%4=12；i==end-1 → slot_end=3*4+5-1*4=13
  - → `[12]`

**关键**：attention 时
```python
if context.block_tables is not None:    # prefix cache
    k, v = k_cache, v_cache
```
因为 C 只需要算 1 个 token，但 attention 要看到全部 5 个，
所以从 KV Cache 读（块0 有 a0-a3）。

模型前向 → a4 的 KV 写到槽 12（块3第0位）。

采样 → C 得到 `t5'`。

### postprocess()

**对 C**：
- `hash_blocks(C)`：`start = 4//4 = 1`, `end = (4+1)//4 = 1` → 相等 → **return**
  （块1还没满）
- `C.num_cached_tokens = 4 + 1 = 5`
- `append_token(t5')` → `num_tokens = 6`
- `num_completion_tokens = 1 != 2` → 继续

### Step 2 结束

```
waiting = []
running = [C]          ← 注意！A、B 不在 running 里
```

⚠️ 等等！A、B 去哪了？

回顾 schedule：**这步只返回了 `[C]`（prefill）**。
但 A、B 明明在 running 队列里啊？让我重新看 schedule 逻辑。

**啊，我漏了一点**：schedule 的 prefill 阶段成功选到序列后，
**直接 `return scheduled_seqs, True`**，**不会走到 decode 阶段**。

所以 A、B **没有被调度**，但它们**仍在 running 队列里**（没被取出）。
`running` 队列此时是 `[A, B]`，加上刚 append 的 C？

回看 prefill 阶段的代码：
```python
if seq.num_cached_tokens + seq.num_scheduled_tokens == seq.num_tokens:
    seq.status = SequenceStatus.RUNNING
    self.waiting.popleft()
    self.running.append(seq)     # ← C 被 append 到 running 队尾
```

所以 running = `[A, B, C]`。A、B 只是**这一步没被调度**，还好好待着。

修正 Step 2 结束状态：

```
waiting = []
running = [A, B, C]

A: num_tokens=6, num_cached_tokens=5, block_table=[0,1]
B: num_tokens=4, num_cached_tokens=3, block_table=[2]
C: num_tokens=6, num_cached_tokens=5, block_table=[0,3]

free = [4,5,6,7]
used = {0,1,2,3}
Block[0].ref_count = 2   ← A 和 C 共享！
```

## 8.5 Step 3：A、B、C 一起 decode

### schedule()

**阶段一：prefill**：`waiting` 空 → 跳过

**阶段二：decode**：

**A**（popleft）：
- `can_append(A)`：`len(A)=6`, `6 % 4 = 2 ≠ 1` → 需要新块？False → 检查 `free >= 0` ✓
- `A.num_scheduled_tokens = 1`, `is_prefill = False`
- `may_append(A)`：`6 % 4 = 2 ≠ 1` → 不分配新块
- 调度 A

**B**（popleft）：
- `can_append(B)`：`len(B)=4`, `4 % 4 = 0 ≠ 1` → ✓
- 调度 B

**C**（popleft）：
- `can_append(C)`：`len(C)=6`, `6 % 4 = 2` → ✓
- 调度 C

**返回 `([A,B,C], False)`**，running 恢复为 `[A,B,C]`。

### run()：decode

`prepare_decode([A,B,C])`：

| 序列 | last_token | positions | context_lens | slot_mapping |
|------|-----------|-----------|--------------|--------------|
| A | t5 | 5 | 6 | `1*4 + (6-1*4) - 1 = 4+2-1 = 5` |
| B | t3 | 3 | 4 | `2*4 + 4 - 1 = 11` |
| C | t5' | 5 | 6 | `3*4 + 2 - 1 = 13` |

`block_tables = [[0,1], [2,-1], [0,3]]`（对齐到 max_len=2，B 补 -1）

KV 写入：
```
槽5  ← A 的 t5    (块1第1位)
槽11 ← B 的 t3    (块2第3位)
槽13 ← C 的 t5'   (块3第1位)
```

`flash_attn_with_kvcache`：
- A: 从块0、1 读 6 个 KV
- B: 从块2 读 4 个 KV
- C: 从块0、3 读 6 个 KV

采样：
- A → `t6`
- B → `t4`
- C → `t6'`

### postprocess()

**A**：
- `hash_blocks(A)`：`start = 5//4 = 1`, `end = (5+1)//4 = 1` → return（块1没满）
- `num_cached_tokens = 6`
- `append_token(t6)` → `num_tokens=7`
- `num_completion_tokens = 7-5 = 2 == max_tokens(2)` → **FINISHED!**
- `deallocate(A)`：
  - 逆序释放块1(ref 1→0，回 free)、块0(ref 2→1，不动)
  - `A.num_cached_tokens = 0`, `block_table = []`
- `running.remove(A)`

**B**：
- `hash_blocks(B)`：`start = 3//4 = 0`, `end = 4//4 = 1` → 登记块2！
  - `h_B0 = compute_hash([b0,b1,b2,t3], -1)`
  - `hash_to_block_id[h_B0] = 2`
- `num_cached_tokens = 4`
- `append_token(t4)` → `num_tokens=5`
- `num_completion_tokens = 5-3 = 2 != 3` → 继续

**C**：
- `hash_blocks(C)`：`start = 5//4 = 1`, `end = 6//4 = 1` → return
- `num_cached_tokens = 6`
- `append_token(t6')` → `num_tokens=7`
- `num_completion_tokens = 2 == 2` → **FINISHED!**
- `deallocate(C)`：块3(ref 1→0，回 free)、块0(ref 1→0，回 free)
- `running.remove(C)`

### Step 3 结束

```
waiting = []
running = [B]

A: FINISHED, 输出 [t5, t6]
B: num_tokens=5, block_table=[2]
C: FINISHED, 输出 [t5', t6']

free = [0,1,3,4,5,6,7]     ← 块0、1、3 都回来了
used = {2}
Block[0].ref_count = 0，但 hash_to_block_id 仍保留 h0→0！
hash_to_block_id = {h0: 0, h_B0: 2}
```

🎉 **关键观察**：块0 的哈希记录 `h0→0` **仍然存在**，
尽管块0已经回到 free 池。这意味着后续如果有请求前缀是 `[a0,a1,a2,a3]`，
还能命中！

## 8.6 Step 4：B 的最后一次 decode

### schedule()

prefill 空 → decode：

**B**：
- `can_append(B)`：`len(B)=5`, `5 % 4 = 1` ✓ **需要新块** → 检查 `free >= 1` ✓
- `may_append(B)`：`5 % 4 == 1` → 分配新块！（从 free 取块0？不，free 是队列，取队首）
  - free = [0,1,3,4,5,6,7]，队首 0 → `_allocate_block()` 取块0
  - ⚠️ 块0 的 hash 记录 `h0→0` 被**删除**（因为内容将被覆盖）
  - `B.block_table = [2, 0]`
- 调度 B

**返回 `([B], False)`**

### run()

- `B.last_token = t4`, `positions = 4`, `context_lens = 5`
- `slot_mapping = 0*4 + (5 - 1*4) - 1 = 0 + 1 - 1 = 0`  ← 写块0的槽0

采样 → `t5`。

### postprocess()

- `hash_blocks(B)`：`start = 4//4 = 1`, `end = 5//4 = 1` → return
- `num_cached_tokens = 5`
- `append_token(t5)` → `num_tokens = 6`
- `num_completion_tokens = 6 - 3 = 3 == 3` → **FINISHED!**
- `deallocate(B)`：块0(ref 1→0，回 free)、块2(ref 1→0，回 free)

### Step 4 结束

```
waiting = []
running = []
→ is_finished() = True！主循环退出
```

## 8.7 输出

`outputs` 字典按 `seq_id` 收集了：

```python
{0: [t5, t6], 1: [t3, t4, t5], 2: [t5', t6']}
```

排序后：
```python
[[t5, t6], [t3, t4, t5], [t5', t6']]
```

解码成文本：
```python
[
  {"text": decode([t5, t6]), "token_ids": [t5, t6]},
  {"text": decode([t3, t4, t5]), "token_ids": [t3, t4, t5]},
  {"text": decode([t5', t6']), "token_ids": [t5', t6']},
]
```

## 8.8 全程状态演进表

| Step | 阶段 | waiting | running | 关键事件 |
|------|------|---------|---------|----------|
| 初始 | — | A,B,C | — | 三个请求入队 |
| 1 | prefill | C | A,B | A、B prefill；**C 因预算不够被推迟** |
| 2 | prefill | — | A,B,C | C prefill，**命中 A 的块0**（前缀缓存！）|
| 3 | decode | — | B | A、C 结束并释放块；B 继续 |
| 4 | decode | — | — | B 结束，全部完成 |

## 8.9 这个走查揭示了什么

### 1. 前缀缓存的时序微妙性

Step 1 里 C 和 A 前缀完全相同，但**没能命中**——
因为 A 的块 hash 要等 postprocess 才登记。C 在同一个 step 被调度时
`hash_to_block_id` 还空着。

**只有在后续 step 才能命中**。这是个容易被忽略的时序细节。

### 2. 共享块的引用计数

Step 2 后 `Block[0].ref_count = 2`（A 和 C 共享）。
A 在 Step 3 结束时 `ref_count` 减到 1，块**不释放**（C 还在用）。
C 结束时减到 0 才真正回 free 池。

### 3. 释放的块仍保留哈希

Step 3 后块0 `ref_count=0` 回到 free 池，但 `hash_to_block_id[h0]=0` **仍在**。
如果此时来个新请求带相同前缀，还能"复活"块0。

### 4. 块被重用时哈希失效

Step 4 里 B 需要新块，取到了块0。此时 `_allocate_block` 里
`del hash_to_block_id[block.hash]` 把 `h0→0` 删掉。
**前缀缓存的"软"性质**：谁先用谁占，用满就挤掉。

### 5. prefill 会耽搁 decode

Step 2 只跑了 C 的 prefill，A、B 的 decode 被推迟一步。
这是"prefill 优先"策略的直接代价。真实 vLLM 有更细致的混合调度，
Nano-vLLM 选择简单。

### 6. 块满时的扩块时机

Step 4 B 从 5 个 token 变 6 个。`len(B)=5` 时 `5%4==1` 触发 `may_append`。
这验证了第 4 章讲的判据：**当 KV 位置数达到块边界时需扩块**。

### 7. 只允许第一个序列分块的影响

Step 1 里 C（5 token）因为 `remaining=4 < 5` 且不是第一个，被推迟。
如果允许任意分块，C 可以在 Step 1 处理 4 个 token。
Nano-vLLM 的简化让 C 多等了一步。

## 8.10 用真实参数再想一遍

把玩具参数换成真实的：

```
block_size = 256
max_num_batched_tokens = 16384
max_num_seqs = 512
```

你会发现：
- 一个 256 token 的单位变成了 256，而非 4
- 前缀缓存的复用粒度是 256 token（不足 256 的部分无法复用）
- 大 batch 时 `max_num_seqs` 和显存才是瓶颈

**物理机制完全一样，只是尺度不同。**

## 8.11 全书回顾

八章走下来，你应该能回答这些问题：

1. **`generate()` 一行背后发生了什么？** → 第 1 章
2. **一条请求的状态如何表达？** → 第 2 章
3. **每一步该跑哪些请求？** → 第 3 章
4. **KV Cache 如何按块复用？前缀缓存怎么工作？** → 第 4 章
5. **逻辑序列如何变成 GPU 张量？CUDA Graph 是什么？** → 第 5 章
6. **KV 怎么存、attention 怎么算？** → 第 6 章
7. **权重如何切分到多卡？** → 第 7 章
8. **三个请求如何走完全程？** → 第 8 章（本章）

## 8.12 延伸阅读

想进一步深入，可以看这些方向：

| 主题 | 建议 |
|------|------|
| 真实 vLLM | 读 vLLM 的 `Scheduler`（支持任意分块、优先级、混合调度）|
| PagedAttention 论文 | Kwon et al. 2023，理解设计动机 |
| FlashAttention | Dao et al.，理解 IO-aware 的 attention |
| Continuous Batching | Orca 论文 |
| CUDA Graph | PyTorch 官方文档的 `torch.cuda.graph` |
| 张量并行 | Megatron-LM 论文 |

## 8.13 结语

Nano-vLLM 用约 1200 行实现了现代推理引擎的核心机制：
PagedAttention、前缀缓存、连续批处理、分块 prefill、抢占、
CUDA Graph、张量并行、flash-attn。

它的价值不在性能极致，而在于**用最小的代码暴露每一层的设计权衡**。
读完本书，再去读 vLLM 的源码，你会认出每一个概念——
它们只是被更复杂的工程细节包裹起来的同一批思想。

---

**🎓 全书完**

回到 [目录](README.md) | 上一章：[模型与张量并行](07-model-tp.md)
