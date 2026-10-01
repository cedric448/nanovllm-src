# 第 2 章 Sequence：一条请求的状态载体

> 本章目标：理解"一条请求"在引擎内部长什么样，以及它携带哪些状态。
> 这是读懂调度器和显存管理的前提。

## 2.1 为什么需要 Sequence 这个类？

用户传入的是一个字符串（或 token 列表）。但在引擎内部，一条请求需要携带**远不止**这些：

- 已经生成了多少 token？
- 有多少 token 的 KV 已经算过并缓存了？（决定下一步从哪开始算）
- 这一步要被调度多少 token？（prefill 分块时需要）
- KV Cache 的块表是什么？（存在哪些 block 里）
- 采样参数是什么？

把这些打包成一个对象，就是 `Sequence`。整个引擎里，**所有模块都以 Sequence 为单位通信**。

## 2.2 状态机：Sequence 的三态

`nanovllm/engine/sequence.py:8`：

```python
class SequenceStatus(Enum):
    WAITING = auto()
    RUNNING = auto()
    FINISHED = auto()
```

三个状态的转换关系：

```
                 add_request()
                      │
                      ▼
              ┌─────────────┐
              │   WAITING   │◄─────────────┐
              └─────────────┘              │
                      │ 被调度做 prefill    │ preempt()
                      │ (prompt 全算完)     │ (显存不够被踢出)
                      ▼                    │
              ┌─────────────┐              │
              │   RUNNING   │──────────────┘
              └─────────────┘
                      │ 遇到 EOS 或达 max_tokens
                      ▼
              ┌─────────────┐
              │  FINISHED   │
              └─────────────┘
```

- **WAITING**：还在 waiting 队列里等调度。
- **RUNNING**：prompt 已经处理完，正在逐 token 生成。
- **FINISHED**：结束了，结果可以取走。

> ⚠️ 易错点：一个正在被 **chunked prefill** 分块处理的序列，
> 每算完一块会**回到 WAITING**（因为它还没算完 prompt）。
> 只有 prompt 全部算完的标志是
> `num_cached_tokens + num_scheduled_tokens == num_tokens`
> （第 3 章会详述）。所以 WAITING 不代表"从未开始"。

## 2.3 Sequence 的字段全解

`nanovllm/engine/sequence.py:14`：

```python
class Sequence:
    block_size = 256
    counter = count()

    def __init__(self, token_ids: list[int], sampling_params = SamplingParams()):
        self.seq_id = next(Sequence.counter)
        self.status = SequenceStatus.WAITING
        self.token_ids = copy(token_ids)
        self.last_token = token_ids[-1]
        self.num_tokens = len(self.token_ids)
        self.num_prompt_tokens = len(token_ids)
        self.num_cached_tokens = 0
        self.num_scheduled_tokens = 0
        self.is_prefill = True
        self.block_table = []
        self.temperature = sampling_params.temperature
        self.max_tokens = sampling_params.max_tokens
        self.ignore_eos = sampling_params.ignore_eos
```

### 类属性

| 属性 | 值 | 说明 |
|------|-----|------|
| `block_size` | 256 | 每个 KV 块装多少 token。由 `LLMEngine.__init__` 从 config 回填 |
| `counter` | `itertools.count()` | 全局自增 ID 生成器（类级别共享） |

### 实例字段（按功能分组）

**身份类**

| 字段 | 含义 |
|------|------|
| `seq_id` | 全局唯一 ID，从 counter 取。用于最后把输出排序还原顺序 |
| `status` | 三态状态机的当前状态 |

**内容类**

| 字段 | 含义 |
|------|------|
| `token_ids` | 全部 token（prompt + 已生成） |
| `last_token` | 最后一个 token，decode 时只需要喂它 |
| `num_tokens` | `len(token_ids)` |
| `num_prompt_tokens` | prompt 的长度（不变） |

**调度进度类**（这是最绕的一组，重点理解）

| 字段 | 含义 | 类比 |
|------|------|------|
| `num_cached_tokens` | 已经算完 KV 并缓存好的 token 数 | "已完成的工作量" |
| `num_scheduled_tokens` | *本次* step 要处理的 token 数 | "本批次要干的活" |
| `is_prefill` | 标记当前是不是 prefill | "工作类型" |

**显存类**

| 字段 | 含义 |
|------|------|
| `block_table` | KV 块表，列表形式：第 i 个元素是第 i 块的 block_id |

**采样类**

| 字段 | 含义 |
|------|------|
| `temperature` | 采样温度 |
| `max_tokens` | 最大生成数 |
| `ignore_eos` | 是否忽略 EOS |

### 为什么 `num_prompt_tokens` 要单独存？

因为要区分 prompt 和生成内容：

```python
@property
def num_completion_tokens(self):
    return self.num_tokens - self.num_prompt_tokens
```

`num_tokens` 会随着生成不断增长，但 `num_prompt_tokens` 是固定的。
这样算生成数只要一次减法，不用反复记录。

### 为什么用 `copy(token_ids)`？

```python
self.token_ids = copy(token_ids)
```

浅拷贝，防止外部修改（比如用户传入的列表）影响引擎内部状态。
`bench.py` 里传的是 `prompt_token_ids` 列表，如果不 copy，
序列推进时会 `append_token` 污染用户的原始数据。

## 2.4 三个核心概念：靠一个例子讲透

围绕 `num_tokens`、`num_cached_tokens`、`num_scheduled_tokens`，
初学者最容易晕。用一个具体例子走一遍。

### 场景设定

- prompt = 600 个 token
- `block_size = 256`
- `max_num_batched_tokens = 1024`

### 朴素猜测（错）

很多人以为：600 个 prompt token，一次 prefill 算完。
但如果预算紧张（比如同一批还有别的序列），可能得分两次。

### 真实流程（对）

**Step 1（prefill 第一块）**

此时：
- `num_tokens = 600`（prompt 长度）
- `num_cached_tokens = 0`（还没算任何 KV）
- `is_prefill = True`

调度器决定这一步处理 `num_scheduled_tokens = 600`（假设预算够）。
算完后 `postprocess` 更新：
- `num_cached_tokens += num_scheduled_tokens` → `600`
- 检查 `num_cached_tokens (600) == num_tokens (600)` → 算完了！
- 取最后一个位置的 logits 采样 → 得到 1 个新 token
- `append_token` → `num_tokens` 变成 `601`

此时序列可以进入 RUNNING。

**Step 2（decode）**

- `num_tokens = 601`
- `num_cached_tokens = 600`
- `is_prefill = False`
- 只要处理 1 个 token（`num_scheduled_tokens = 1`）：即第 601 个 token 的 attention

算完后：
- `num_cached_tokens = 601`
- 采样出新 token → `num_tokens = 602`

**Step 3、4、5...** 重复 Step 2，每次 `num_tokens` 和 `num_cached_tokens` 都 +1。

### 分块 prefill 的变体

假设预算只够 400：

**Step 1**：`num_scheduled_tokens = 400`，算完后
- `num_cached_tokens = 400`
- `400 != 600` → 还没算完 → **序列回到 WAITING**

**Step 2**：`num_scheduled_tokens = 200`，算完后
- `num_cached_tokens = 600`
- `600 == 600` → 算完 → 采样第一个新 token → 进入 RUNNING

这就是为什么第 3 章的 schedule 里有一句：
```python
if seq.num_cached_tokens + seq.num_scheduled_tokens == seq.num_tokens:
    seq.status = SequenceStatus.RUNNING
```

### 一张演进表

| 时刻 | num_tokens | num_cached_tokens | num_scheduled_tokens | is_prefill | status |
|------|-----------|-------------------|---------------------|-----------|--------|
| 初始 | 600 | 0 | 0 | True | WAITING |
| Step1 调度后 | 600 | 0 | 600 | True | WAITING |
| Step1 完成后 | 601 | 600 | 0 | True | RUNNING |
| Step2 调度后 | 601 | 600 | 1 | False | RUNNING |
| Step2 完成后 | 602 | 601 | 0 | False | RUNNING |
| ... | ... | ... | ... | ... | ... |
| 遇 EOS 后 | 650 | 649 | 0 | False | FINISHED |

## 2.5 所有魔法方法

### `__len__` 与 `__getitem__`

```python
def __len__(self):
    return self.num_tokens

def __getitem__(self, key):
    return self.token_ids[key]
```

这让 `Sequence` 用起来像个列表：

```python
len(seq)           # 等价于 seq.num_tokens
seq[5]             # 等价于 seq.token_ids[5]
seq[10:20]         # 切片也能用！
```

⚠️ 注意 `len(seq)` 和 `len(seq.token_ids)` 是等价的，但**前者更常用**，
因为 `num_tokens` 是显式字段，不用每次重算。

### 各种 `@property`

```python
@property
def is_finished(self):
    return self.status == SequenceStatus.FINISHED

@property
def num_completion_tokens(self):
    return self.num_tokens - self.num_prompt_tokens

@property
def prompt_token_ids(self):
    return self.token_ids[:self.num_prompt_tokens]

@property
def completion_token_ids(self):
    return self.token_ids[self.num_prompt_tokens:]

@property
def num_blocks(self):
    return (self.num_tokens + self.block_size - 1) // self.block_size

@property
def last_block_num_tokens(self):
    return self.num_tokens - (self.num_blocks - 1) * self.block_size
```

逐个解释：

- **`is_finished`**：状态是否 FINISHED。引擎用它判断能否产出结果。
- **`num_completion_tokens`**：已生成多少 token。
- **`prompt_token_ids` / `completion_token_ids`**：切出 prompt 和生成部分。
  `generate()` 最后返回的 `token_ids` 就是 `completion_token_ids`。
- **`num_blocks`**：需要多少个块。向上取整公式
  `(n + block_size - 1) // block_size`。
  例：n=600、block_size=256 → `(600+255)//256 = 3` 块。
  为什么是 3 而不是 2？因为 600 > 512，需要第 3 块装剩下的 88 个。
- **`last_block_num_tokens`**：最后一个块里装了多少 token。
  例：600 → `600 - 2×256 = 88`。
  decode 写 KV 时要用它算 slot 位置（第 6 章）。

### `block(i)`：取第 i 块的 token

```python
def block(self, i):
    assert 0 <= i < self.num_blocks
    return self.token_ids[i*self.block_size: (i+1)*self.block_size]
```

例：`token_ids` 有 600 个、block_size=256：
- `block(0)` → 前 256 个
- `block(1)` → 第 256~511 个
- `block(2)` → 第 512~599 个（只有 88 个）

**这个方法是 Prefix Caching 的基础**——比较两个序列的 `block(i)` 是否相同，
就能判断前缀是否一致（第 4 章）。

### `append_token`：生成新 token

```python
def append_token(self, token_id: int):
    self.token_ids.append(token_id)
    self.last_token = token_id
    self.num_tokens += 1
```

三件事：追加、更新 last_token、更新计数。
decode 每步结束后调用一次。

## 2.6 序列化：为多进程准备

这是 Sequence 里**最隐晦**的一段。先看代码：

```python
def __getstate__(self):
    last_state = self.last_token if not self.is_prefill else self.token_ids
    return (self.num_tokens, self.num_prompt_tokens, self.num_cached_tokens,
            self.num_scheduled_tokens, self.block_table, last_state)

def __setstate__(self, state):
    self.num_tokens, self.num_prompt_tokens, self.num_cached_tokens, \
    self.num_scheduled_tokens, self.block_table, last_state = state
    if isinstance(last_state, list):
        self.token_ids = last_state
        self.last_token = self.token_ids[-1]
    else:
        self.token_ids = []
        self.last_token = last_state
```

### 为什么需要这个？

张量并行（多卡）时，rank=0 要把 `Sequence` 对象通过**共享内存 pickle**
发送给 rank>0 的子进程（第 5 章）。Python 的 pickle 默认会序列化
`__dict__` 里所有字段，包括整个 `token_ids` 列表。

问题：**解码阶段的 token_ids 可能很长**（几千个），
但子进程其实**只需要最后一个 token**（decode 只喂 last_token）。
每次都把完整列表 pickle 过去，纯属浪费带宽。

### 这个优化怎么做的？

`__getstate__` 里根据 `is_prefill` 决定发什么：

```python
last_state = self.last_token if not self.is_prefill else self.token_ids
```

- **prefill 时**：发完整的 `token_ids`（因为要处理整段 prompt）
- **decode 时**：只发 `last_token`（一个整数！）

`__setstate__` 逆向还原：
```python
if isinstance(last_state, list):
    self.token_ids = last_state          # prefill：整段都在
    self.last_token = self.token_ids[-1]
else:
    self.token_ids = []                  # decode：本地不需要完整列表
    self.last_token = last_state
```

### 关键观察：子进程不持有完整 token_ids

decode 时子进程的 `self.token_ids` 是**空列表**，只有 `last_token`。

那子进程怎么算 attention？
——靠 KV Cache！decode 的 attention 只需要当前 token 的 q 和整段历史的 KV，
而历史 KV 已经在缓存里了（通过 block_table 定位），**不需要重新知道历史 token 是什么**。

这是一个精妙的省流设计。理解这点，才能理解第 5 章
`prepare_decode` 为什么只用 `seq.last_token`。

### 状态图

```
rank=0 (主进程)                     rank=1..N (子进程)
  token_ids = [t0..t599]             token_ids = []            (decode时)
  last_token = t599      ──pickle──► last_token = t599
  block_table = [3,7,9]               block_table = [3,7,9]    ← 必须传！用于定位KV
  num_tokens = 600                    num_tokens = 600
```

⚠️ **注意 `block_table` 必须完整传递**。因为两个进程的 KV Cache
是各自独立分配的（每张卡存自己那份 head 的 KV），
但**块编号必须一致**，这样 rank0 算出的 slot_mapping 对每个 rank 都成立。

## 2.7 完整实例：跟踪一个 Sequence 的一生

```python
# 用户调用
llm.generate(["今天天气"], SamplingParams(temperature=0.6, max_tokens=3))
```

假设 "今天天气" 编码成 `[101, 2345, 678, 90]`（4 个 token），EOS=151643。

### 创建时刻

```python
seq = Sequence([101, 2345, 678, 90], SamplingParams(temperature=0.6, max_tokens=3))
```

状态：
```
seq_id = 0
status = WAITING
token_ids = [101, 2345, 678, 90]
last_token = 90
num_tokens = 4
num_prompt_tokens = 4
num_cached_tokens = 0
num_scheduled_tokens = 0
is_prefill = True
block_table = []
num_blocks = (4 + 255) // 256 = 1
```

### 第一次 schedule（prefill）

调度器分配 1 个块（块号假设是 5），设置 `num_scheduled_tokens = 4`。

```python
block_table = [5]
num_scheduled_tokens = 4
```

模型前向 → 4 个位置都出 logits，**只取最后一个位置**（第 4 个）采样。
假设采到 token `8888`。

### postprocess

```python
num_cached_tokens += 4       # → 4
num_cached_tokens == num_tokens  # 4 == 4 ✓ 算完了
append_token(8888)           # num_tokens → 5, last_token → 8888
# 8888 != EOS(151643), num_completion_tokens(1) != max_tokens(3) → 继续
status = RUNNING
```

现在：
```
num_tokens = 5
num_cached_tokens = 4
num_completion_tokens = 5 - 4 = 1
last_block_num_tokens = 5 - 0*256 = 5
```

### 第二次 schedule（decode）

```python
num_scheduled_tokens = 1
is_prefill = False
```

模型只喂 `last_token=8888`，用 block_table=[5] 定位历史 KV，
算 attention → 采样 → 假设得到 `9999`。

postprocess：
```python
num_cached_tokens += 1  # → 5
append_token(9999)      # num_tokens → 6
num_completion_tokens = 2  # 还不够
```

### 第三次 schedule（decode）

采到 EOS `151643`。

postprocess：
```python
append_token(151643)
num_completion_tokens == 3 == max_tokens → FINISHED
# 或者 token_id == eos → FINISHED
block_manager.deallocate(seq)  # 释放块 5
```

### 输出

```python
seq.completion_token_ids = token_ids[4:] = [8888, 9999, 151643]
tokenizer.decode([8888, 9999, 151643]) = "晴转多云。" (举例)
```

最终返回 `{"text": "晴转多云。", "token_ids": [8888, 9999, 151643]}`。

## 2.8 本章小结

- `Sequence` 是一条请求的完整状态载体，引擎内所有模块以它为单位通信。
- 三态状态机：WAITING → RUNNING → FINISHED（可能因 preempt 回 WAITING）。
- **三个关键计数**：`num_tokens`（总长）、`num_cached_tokens`（已算 KV）、
  `num_scheduled_tokens`（本步要算的）。
  `num_cached_tokens + num_scheduled_tokens == num_tokens` 是"prompt 算完"的判据。
- `block_table` 是逻辑块表，配合 BlockManager 实现分页（第 4 章）。
- `__getstate__`/`__setstate__` 通过"decode 只传 last_token"大幅减少进程间通信量。

**下一章**：[调度器](03-scheduler.md)——引擎的大脑，决定每一步跑谁。
