# 第 3 章 Scheduler：引擎的大脑

> 本章目标：理解"每一步该跑哪些请求、各跑多少 token"这个决策是怎么做出的。
> 这是整个引擎最核心、也最容易读晕的部分。

## 3.1 调度器要解决什么问题

回顾第 0 章的痛点：GPU 是极度并行的设备，但请求是动态到达、长短不一的。
调度器的任务就是**在每一步，从所有待处理请求中挑出一批，喂给 GPU**，使得：

1. GPU 尽可能满载（吞吐高）；
2. 每个请求的等待时间尽可能短（延迟低）；
3. 显存不超（不能 OOM）。

这三个目标会冲突，调度策略就是一种权衡。

## 3.2 两个队列：waiting 与 running

`nanovllm/engine/scheduler.py:10`：

```python
class Scheduler:

    def __init__(self, config: Config):
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.eos = config.eos
        self.block_size = config.kvcache_block_size
        self.block_manager = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size)
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()
```

调度器持有两个 `deque`（双端队列）：

| 队列 | 装着什么 | 操作 |
|------|----------|------|
| `waiting` | 还没开始 / 被抢占 / prefill 没算完的序列 | 从左侧取（FIFO），`popleft` |
| `running` | prompt 已算完、正在逐 token 生成的序列 | 从左侧取，`popleft`；必要时从右侧踢，`pop` |

### 为什么用 deque 而不是 list？

因为 `waiting` 经常需要在**队首插入**（preempt 把序列塞回去，`appendleft`），
deque 两端操作都是 O(1)，而 list 的头部插入是 O(n)。

调度器还持有 `block_manager`——注意它是在调度器构造时创建的，
说明**显存管理和调度是紧耦合的**：调度时就要判断显存够不够。

### `is_finished`

```python
def is_finished(self):
    return not self.waiting and not self.running
```

两个队列都空 = 所有请求处理完。这就是 `LLMEngine.generate()` 主循环的终止条件。

## 3.3 `schedule()` 全貌

`nanovllm/engine/scheduler.py:25`，这是全书最需要反复读的函数：

```python
def schedule(self) -> tuple[list[Sequence], bool]:
    scheduled_seqs = []
    num_batched_tokens = 0

    # ===== 阶段一：prefill =====
    while self.waiting and len(scheduled_seqs) < self.max_num_seqs:
        seq = self.waiting[0]
        remaining = self.max_num_batched_tokens - num_batched_tokens
        if remaining == 0:
            break
        if not seq.block_table:
            num_cached_blocks = self.block_manager.can_allocate(seq)
            if num_cached_blocks == -1:
                break
            num_tokens = seq.num_tokens - num_cached_blocks * self.block_size
        else:
            num_tokens = seq.num_tokens - seq.num_cached_tokens
        if remaining < num_tokens and scheduled_seqs:  # only allow chunked prefill for the first seq
            break
        if not seq.block_table:
            self.block_manager.allocate(seq, num_cached_blocks)
        seq.num_scheduled_tokens = min(num_tokens, remaining)
        num_batched_tokens += seq.num_scheduled_tokens
        if seq.num_cached_tokens + seq.num_scheduled_tokens == seq.num_tokens:
            seq.status = SequenceStatus.RUNNING
            self.waiting.popleft()
            self.running.append(seq)
        scheduled_seqs.append(seq)

    if scheduled_seqs:
        return scheduled_seqs, True

    # ===== 阶段二：decode =====
    while self.running and len(scheduled_seqs) < self.max_num_seqs:
        seq = self.running.popleft()
        while not self.block_manager.can_append(seq):
            if self.running:
                self.preempt(self.running.pop())
            else:
                self.preempt(seq)
                break
        else:
            seq.num_scheduled_tokens = 1
            seq.is_prefill = False
            self.block_manager.may_append(seq)
            scheduled_seqs.append(seq)
    assert scheduled_seqs
    self.running.extendleft(reversed(scheduled_seqs))
    return scheduled_seqs, False
```

结构非常清晰：**两个 while 循环，prefill 优先，prefill 没东西可跑才 decode**。

## 3.4 阶段一：Prefill 调度逐行精讲

### 循环条件

```python
while self.waiting and len(scheduled_seqs) < self.max_num_seqs:
```

两个退出条件：
- `waiting` 空了（没人要做 prefill 了）
- 已选序列达到了 `max_num_seqs` 上限

### 看一眼队首

```python
seq = self.waiting[0]
```

注意是**看一眼**（`[0]`）不是取出来（`popleft`）。
因为可能因为预算不够而不选它，那就得原样留着。

### 算剩余预算

```python
remaining = self.max_num_batched_tokens - num_batched_tokens
if remaining == 0:
    break
```

`num_batched_tokens` 是本步已分配的 token 数。
预算耗尽就停止选新序列。

### 决定这个序列要处理多少 token

```python
if not seq.block_table:
    num_cached_blocks = self.block_manager.can_allocate(seq)
    if num_cached_blocks == -1:
        break
    num_tokens = seq.num_tokens - num_cached_blocks * self.block_size
else:
    num_tokens = seq.num_tokens - seq.num_cached_tokens
```

分两种情况：

**情况 A：`block_table` 为空**（全新序列，或上次被 deallocate 了）

说明这个序列还没分配过 KV 块。要先问显存管理器"能分配吗"：
- `can_allocate(seq)` 返回能复用的**前缀块数** `num_cached_blocks`，
  或者 `-1`（显存不够，第 4 章详解）
- `-1` → `break`，这个序列这步不跑了（等显存释放）

计算需要处理的 token 数：
```python
num_tokens = seq.num_tokens - num_cached_blocks * self.block_size
```
含义：**总长 减去 已命中前缀缓存的长度**。
例：prompt 600 token，命中 2 个前缀块（512 token）→ 只需算 `600 - 512 = 88` 个 token。

**情况 B：`block_table` 非空**（chunked prefill 的后续块，或之前已经分配过）

```python
num_tokens = seq.num_tokens - seq.num_cached_tokens
```
含义：**总长 减去 已经算过的部分**。这就是 chunked prefill 的"剩余待处理量"。

### 关键判断：只允许第一个序列分块

```python
if remaining < num_tokens and scheduled_seqs:  # only allow chunked prefill for the first seq
    break
```

这行代码是本实现的一个**重要简化**。逐字翻译：

- `remaining < num_tokens`：剩余预算不够处理完这个序列
- `and scheduled_seqs`：且已经选了至少一个序列（说明这**不是**第一个）

→ 则 `break`，不选它。

**含义**：只有**第一个被调度的序列**允许被切成块（chunked），
后面进来的序列必须能一次算完，否则不选。

**为什么这样简化？**

真正的 vLLM 允许任意序列分块。但那样需要更复杂的 bookkeeping：
每个序列的 chunk 状态、多序列交错分块、跨 step 的等待队列排序等。

Nano-vLLM 为了代码简洁，选择：第一个序列可以很长（分块慢慢啃），
其余序列要么短到能一次算完，要么等着。

**副作用**：如果队首是个很长的序列，而它后面的序列也很长，
那后面的序列会一直堵塞（队首堵路）。

### 分配并记录

```python
if not seq.block_table:
    self.block_manager.allocate(seq, num_cached_blocks)
seq.num_scheduled_tokens = min(num_tokens, remaining)
num_batched_tokens += seq.num_scheduled_tokens
```

- 首次见到这个序列才真正分配块（`allocate`）
- `num_scheduled_tokens` 取 `num_tokens` 和 `remaining` 的较小值
  （第一个序列可能被 `remaining` 截断，形成分块）
- 累加预算消耗

### 判断是否完成 prefill

```python
if seq.num_cached_tokens + seq.num_scheduled_tokens == seq.num_tokens:
    seq.status = SequenceStatus.RUNNING
    self.waiting.popleft()
    self.running.append(seq)
scheduled_seqs.append(seq)
```

这就是第 2 章讲的判据。如果**本次调度后能算完 prompt**：
- 状态转 RUNNING
- 从 waiting 移到 running
- 否则**留在 waiting**（下一步继续分块）

最后无论完成与否，都加入 `scheduled_seqs`（本步要跑它）。

### prefill 阶段的返回值

```python
if scheduled_seqs:
    return scheduled_seqs, True
```

只要选了序列就返回 `is_prefill=True`。**注意：prefill 阶段哪怕选了
一个还没算完的序列，也是 prefill 模式**。

## 3.5 阶段二：Decode 调度逐行精讲

只有 prefill 阶段什么都没选到（`scheduled_seqs` 为空）才会走到这。

### 取一个 running 序列

```python
while self.running and len(scheduled_seqs) < self.max_num_seqs:
    seq = self.running.popleft()
```

从 running 队首取。

### 检查能否追加 KV，不能则抢占

```python
while not self.block_manager.can_append(seq):
    if self.running:
        self.preempt(self.running.pop())
    else:
        self.preempt(seq)
        break
else:
    seq.num_scheduled_tokens = 1
    seq.is_prefill = False
    self.block_manager.may_append(seq)
    scheduled_seqs.append(seq)
```

这段是 **Python 的 while-else 语法**，很绕，必须讲清：

- `while` 条件为假时（`can_append` 为真）执行 `else` 块；
- `while` 里 `break` 了则**跳过** `else` 块。

翻译成自然语言：

1. 如果 `can_append(seq)`（能追加 KV）→ 条件为假，直接执行 else：
   调度它跑 1 个 token，标记为 decode，maybe 扩块，加入队伍。

2. 如果 `can_append(seq)` 为假（显存不够啦）：
   - 如果 running 里还有别的序列 → **抢占队尾那个**（`self.running.pop()`），
     释放它的块，再试试当前 seq 能不能 append；
   - 如果 running 里只剩当前这个 → **抢占它自己**，`break` 跳过 else，
     这个序列这步就不跑了（回到 waiting 等地）。

### 什么是 preempt（抢占）？

`nanovllm/engine/scheduler.py:75`：

```python
def preempt(self, seq: Sequence):
    seq.status = SequenceStatus.WAITING
    seq.is_prefill = True
    self.block_manager.deallocate(seq)
    self.waiting.appendleft(seq)
```

三件事：
1. 状态改回 WAITING
2. **`is_prefill = True`** ← 重点！
3. 释放它的 KV 块
4. 塞回 waiting 队首（优先重新调度）

**为什么被抢占后要标记 `is_prefill = True`？**

因为它的 KV 已经被释放了。当它重新被调度时，必须**重新 prefill**——
从头把 prompt 的 KV 再算一遍（虽然可以靠前缀缓存的 hash 命中恢复一部分）。

⚠️ 这是一个重要事实：**抢占是有代价的**。被抢占的序列要重算 KV，
而不是接着 decode。所以调度器应尽量避免抢占（这也是为什么优先牺牲队尾）。

### 把调度结果放回 running

```python
assert scheduled_seqs
self.running.extendleft(reversed(scheduled_seqs))
return scheduled_seqs, False
```

`extendleft(reversed(x))` 的效果是把 `x` **原序**放到队首左侧。
比如 running 原本是 `[C, D]`，scheduled 是 `[A, B]`：
```
extendleft(reversed([A,B])) = extendleft([B,A])
→ 依次 appendleft B、appendleft A → running = [A, B, C, D] ✓
```
保持调度顺序不变，且被调度的排前面。

`assert scheduled_seqs`：decode 阶段至少要能调度一个序列，
否则说明代码逻辑有 bug（因为 `is_finished` 为假就一定有 running 序列）。

## 3.6 `postprocess()`：写回结果

`nanovllm/engine/scheduler.py:81`：

```python
def postprocess(self, seqs: list[Sequence], token_ids: list[int], is_prefill: bool):
    for seq, token_id in zip(seqs, token_ids):
        self.block_manager.hash_blocks(seq)
        seq.num_cached_tokens += seq.num_scheduled_tokens
        seq.num_scheduled_tokens = 0
        if is_prefill and seq.num_cached_tokens < seq.num_tokens:
            continue
        seq.append_token(token_id)
        if (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens == seq.max_tokens:
            seq.status = SequenceStatus.FINISHED
            self.block_manager.deallocate(seq)
            self.running.remove(seq)
```

逐行：

### 记录块的哈希（前缀缓存用）

```python
self.block_manager.hash_blocks(seq)
```
把本步新填满的块计算哈希，登记到全局哈希表（第 4 章）。**必须在更新计数前调用**，
因为它需要知道"哪段 token 刚被算完"。

### 更新计数

```python
seq.num_cached_tokens += seq.num_scheduled_tokens
seq.num_scheduled_tokens = 0
```
本步处理过的 token 现在算作"已缓存"了，清零 scheduled。

### 分块 prefill 的中间状态：不采样

```python
if is_prefill and seq.num_cached_tokens < seq.num_tokens:
    continue
```

**这是 chunked prefill 的关键**：如果这步是 prefill 且 prompt 还没算完，
说明这是个中间块，**还不能采样**（还没算到 prompt 末尾），直接 `continue`。

⚠️ 这也意味着：`token_ids` 里对应这个 seq 的位置可能是占位/无效值，
反正会被 `continue` 跳过。

### 追加 token 并判断结束

```python
seq.append_token(token_id)
if (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens == seq.max_tokens:
    seq.status = SequenceStatus.FINISHED
    self.block_manager.deallocate(seq)
    self.running.remove(seq)
```

两个终止条件（或关系）：
1. 没设 `ignore_eos` 且采到了 EOS
2. 生成数达到 `max_tokens`

终止时释放 KV 块，并从 running 队列移出。
（`deallocate` 会把块还给空闲池，别的序列可以复用。）

## 3.7 完整实例：分块 prefill + 抢占

用一个精心设计的例子把所有机制串起来。

### 场景

- `max_num_batched_tokens = 1024`
- `max_num_seqs = 4`
- `block_size = 256`
- 序列：
  - A：prompt = 1500 token（长！要分块）
  - B：prompt = 100 token
  - C：prompt = 100 token

### Step 1：调度 A

队首是 A。`num_tokens = 1500 - 0 = 1500`。
`remaining = 1024`。`1500 > 1024` 但 `scheduled_seqs` 为空（A 是第一个），
所以**允许分块**。

```
A.num_scheduled_tokens = min(1500, 1024) = 1024
num_batched_tokens = 1024
A.num_cached_tokens(0) + 1024 = 1024 < 1500 → A 留在 waiting
```

下一个循环：`remaining = 1024 - 1024 = 0` → `break`。

**返回 `([A], True)`**。这步只跑 A 的前 1024 个 token。

```
waiting = [A(剩476), B, C]
```

### Step 2：继续 A

队首还是 A。此时 A 的 `block_table` 非空（Step1 分配过），
所以 `num_tokens = 1500 - 1024 = 476`。

```
A.num_scheduled_tokens = min(476, 1024) = 476
num_batched_tokens = 476
A.num_cached_tokens(1024) + 476 = 1500 == 1500 ✓ 算完了！
→ A.status = RUNNING, 移到 running
```

继续循环，看 B：`remaining = 1024 - 476 = 548`，B 需要 100。

```
B.num_scheduled_tokens = min(100, 548) = 100
num_batched_tokens = 576
B 算完 → RUNNING, 移到 running
```

继续看 C：`remaining = 1024 - 576 = 448`，C 需要 100。

```
C.num_scheduled_tokens = min(100, 448) = 100
C 算完 → RUNNING
```

**返回 `([A, B, C], True)`**。这一步 A 收尾、B、C 全部 prefill 完。

```
waiting = []
running = [A, B, C]
```

### Step 3：Decode 全部

waiting 空 → 进 decode 阶段。

```python
A.num_scheduled_tokens = 1  (如果 can_append)
B.num_scheduled_tokens = 1
C.num_scheduled_tokens = 1
```

**返回 `([A, B, C], False)`**。三个一起 decode。

假设这一步 C 显存告急，`can_append(C)` 为假：

```
C 是 running 队尾？假设 running = [A, B, C]
popleft 取 A → can_append(A) 真 → 调度 A
popleft 取 B → can_append(B) 真 → 调度 B
popleft 取 C → can_append(C) 假 → running 已空 → preempt(C) 自己
  C.status = WAITING, is_prefill = True, deallocate, waiting.appendleft(C)
  break（C 不跑）
```

**返回 `([A, B], False)`**，但注意 `assert scheduled_seqs` 通过（有 A、B）。

```
waiting = [C]  (要重新 prefill!)
running = [A, B]
```

### Step 4：C 重新 prefill

waiting 有 C → 进 prefill 阶段。

C 重新 prefill，但它之前在 Step2 算过的 KV 已经被 deallocate 了。
不过——**前缀缓存可能救它**：如果它的块 hash 还在全局哈希表里，
`can_allocate` 会返回命中的块数，不需要全部重算（第 4 章）。

假设完全没命中，C 从头 100 token 算起。

**返回 `([C], True)`**。

```
running = [A, B, C]  (C 回来了)
```

### 这个例子揭示了什么

1. **长序列优先独占第一步**（A 拿走 1024 预算）
2. **分块 prefill 让长 prompt 不阻塞太久**（A 两个 step 搞定）
3. **抢占是最后手段**（C 被踢出，代价是重算 KV）
4. **抢占后自动重排到队首**，尽快恢复

## 3.8 三种调度模式对比

| 模式 | 触发条件 | 返回 | 序列的 num_scheduled_tokens |
|------|----------|------|----------------------------|
| Prefill | `waiting` 非空 | `(seqs, True)` | 可能 >1（多 token） |
| Chunked Prefill | 队首长 + 预算不足 | `(seqs, True)` | = 剩余预算 |
| Decode | `waiting` 空 | `(seqs, False)` | 恒为 1 |

## 3.9 与真实 vLLM 的差异

作为学习材料，了解这些简化很有价值：

| 特性 | vLLM | Nano-vLLM |
|------|------|-----------|
| 任意序列分块 | ✅ | ❌ 只有第一个 |
| 优先级调度 | ✅ | ❌ 纯 FIFO |
| 抢占策略 | 重算/换出 KV 两种 | 只有重算 |
| 混合 prefill+decode | ✅ | ❌ 分阶段 |
| beam search | ✅ | ❌ |

这些简化让代码从 vLLM 的上万行降到 92 行，同时保留了核心思想。

## 3.10 本章小结

- 调度器维护 `waiting`（待处理）和 `running`（生成中）两个队列。
- `schedule()` **prefill 优先**：先尽力喂饱 prefill 预算，没有才做 decode。
- **只允许第一个序列分块**，这是核心简化，也让队首可能堵路。
- decode 阶段每个序列分配 1 个 token；显存不够时**从队尾开始抢占**。
- 抢占的代价是序列被 `is_prefill=True` 重置，KV 需重算（但可命中前缀缓存）。
- `postprocess()` 更新计数、写回 token、判断结束、释放块。
- chunked prefill 中间块靠 `num_cached_tokens < num_tokens` 跳过采样。

**下一章**：[分页显存管理](04-block-manager.md)——KV Cache 如何按块复用，
Prefix Caching 的秘密就在这里。
