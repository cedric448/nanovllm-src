# 第 4 章 BlockManager：分页显存管理与前缀缓存

> 本章目标：理解 PagedAttention 的"分页"到底是什么，
> 以及 Prefix Caching（前缀缓存）如何用哈希链实现自动复用。

## 4.1 从"连续分配"到"分页"

### 朴素方案：连续 KV Cache

给每个序列分配一段**连续**的显存（比如都按 max_len 预留）：

```
物理显存:  [──A 的 4096 ──][──B 的 4096 ──][──C 的 4096 ──]
实际使用:  [A:10][   空洞   ][B:20][  空洞  ][C:4096 满了]
```

问题：
- **内部碎片**：A 只用 10 个位置，剩下 4086 白占
- **无法共享**：A 和 B 的 prompt 前缀完全相同，也得各存一份

### PagedAttention：像操作系统虚拟内存一样

把 KV Cache 切成固定大小的**块**（Nano-vLLM 里是 256 个 token 一块），
每个序列用一个"**块表**"（block_table）记录它占用了哪些物理块：

```
物理块池:  [块0][块1][块2][块3][块4][块5][块6][块7] ...
              ↑         ↑    ↑         ↑
序列A block_table = [0, 2]         (A 用了块0、块2)
序列B block_table = [1, 5, 6]      (B 用了块1、块5、块6)
```

好处：
- **消除内部碎片**：按需分配，只有最后一个块可能浪费
- **允许共享**：A 和 B 前缀相同 → 指向同一个物理块
- **允许非连续**：逻辑上连续的 token，物理上可以分散

### 与操作系统虚拟内存的类比

| 操作系统的概念 | Nano-vLLM 的对应 |
|---------------|-----------------|
| 页（Page） | 块（Block），256 token |
| 页表（Page Table） | `seq.block_table` |
| 物理页框 | 物理 block（`Block` 对象） |
| 页表项指向共享页 | `ref_count > 1` |
| 缺页 | 前缀缓存未命中，需重算 |

## 4.2 Block 类

`nanovllm/engine/block_manager.py:8`：

```python
class Block:

    def __init__(self, block_id):
        self.block_id = block_id
        self.ref_count = 0
        self.hash = -1
        self.token_ids = []

    def update(self, hash: int, token_ids: list[int]):
        self.hash = hash
        self.token_ids = token_ids

    def reset(self):
        self.ref_count = 1
        self.hash = -1
        self.token_ids = []
```

四个字段：

| 字段 | 类型 | 含义 |
|------|------|------|
| `block_id` | int | 物理块号（索引），从 0 到 num_blocks-1 |
| `ref_count` | int | 引用计数。多少个序列在用这个块（共享前缀时为 2+） |
| `hash` | int | 这个块内容的哈希值（-1 表示未计算）。前缀缓存的 key |
| `token_ids` | list | 这个块装的 token（用于哈希校验，防止哈希碰撞误判） |

### `ref_count` 的作用

这是**共享**的基础。当两个序列前缀相同时：

```
序列A.block_table = [3, 7]
序列B.block_table = [3, 7, 9]     ← 共享块3、块7
Block[3].ref_count = 2
Block[7].ref_count = 2
Block[9].ref_count = 1
```

当 A 结束时，`ref_count` 减到 1，块不释放（B 还在用）。
只有减到 0 才真正释放。

### `token_ids` 为什么要存？

因为**哈希可能碰撞**。两个不同的 token 序列理论上可能哈希到同一个值
（虽然 xxhash64 碰撞概率极低）。存下原始 token，命中时再逐字节比对，
做到绝对安全（见 `can_allocate` 的校验）。

### `reset()`

被复用时调用：把 `ref_count` 设为 1（新主人），清空 hash 和 token_ids。
注意 `ref_count` 设为 1 而不是 0——因为调用 `reset()` 就意味着"已经有一个新序列要用它"。

## 4.3 BlockManager 的字段

`nanovllm/engine/block_manager.py:26`：

```python
class BlockManager:

    def __init__(self, num_blocks: int, block_size: int):
        self.block_size = block_size
        self.blocks: list[Block] = [Block(i) for i in range(num_blocks)]
        self.hash_to_block_id: dict[int, int] = dict()
        self.free_block_ids: deque[int] = deque(range(num_blocks))
        self.used_block_ids: set[int] = set()
```

| 字段 | 含义 |
|------|------|
| `blocks` | 所有物理块的数组，`blocks[i]` 是块号 i 的对象 |
| `hash_to_block_id` | 全局哈希表：`hash → block_id`。前缀缓存的索引 |
| `free_block_ids` | 空闲块号队列 |
| `used_block_ids` | 使用中块号的集合 |

### 三处状态的配合

```
free_block_ids = [4, 5, 6, 7]     ← 可分配
used_block_ids = {0, 1, 2, 3}     ← 已分配
hash_to_block_id = {h1: 0, h2: 1} ← 已缓存的块内容索引
```

⚠️ 注意 `hash_to_block_id` 里的块**可能不在 used 集合里**。
一个块被释放回到 free 池后，它的哈希记录**仍然保留**
（这是前缀缓存能命中的关键）——直到这个块被重新分配、内容被覆盖。

## 4.4 哈希链：前缀缓存的灵魂

`nanovllm/engine/block_manager.py:35`：

```python
@classmethod
def compute_hash(cls, token_ids: list[int], prefix: int = -1):
    h = xxhash.xxh64()
    if prefix != -1:
        h.update(prefix.to_bytes(8, "little"))
    h.update(np.array(token_ids).tobytes())
    return h.intdigest()
```

关键设计：**哈希包含前一个块的哈希**（`prefix` 参数）。

### 为什么要这样？单看代码可能觉得多余

假设我们**不用**前缀哈希（每块独立哈希）：

```
序列A: [你好][世界][今天]        → 块0哈希=h(你好), 块1哈希=h(世界)
序列B: [我][世界][今天]          → 块0哈希=h(我),   块1哈希=h(世界)  ← 撞了！
```

块1的 token 相同（都是"世界"），哈希相同 → 系统误以为 B 的块1
可以复用 A 的块1。但**块1在 A 和 B 里的上下文位置不同**，
attention 的结果完全不同！复用会出大错。

### 用哈希链解决

```
A 块0: h0 = hash([你好], -1)
A 块1: h1 = hash([世界], h0)      ← 包含 h0
A 块2: h2 = hash([今天], h1)

B 块0: h0' = hash([我], -1)       ← 和 h0 不同
B 块1: h1' = hash([世界], h0')    ← 即使 token 相同，前缀不同 → h1' ≠ h1 ✓
B 块2: h2' = hash([今天], h1')
```

这样，**只有从序列开头起每一块都相同**，哈希链才会一路匹配。

```
序列A: [你好][世界][今天]
序列C: [你好][世界][明天]
       ↑ 块0匹配  ↑ 块1匹配(hash含h0)  ↑ 块2不同 → 只复用前2块
```

## 4.5 `can_allocate()`：探测能复用多少

`nanovllm/engine/block_manager.py:58`：

```python
def can_allocate(self, seq: Sequence) -> int:
    h = -1
    num_cached_blocks = 0
    num_new_blocks = seq.num_blocks
    for i in range(seq.num_blocks - 1):
        token_ids = seq.block(i)
        h = self.compute_hash(token_ids, h)
        block_id = self.hash_to_block_id.get(h, -1)
        if block_id == -1 or self.blocks[block_id].token_ids != token_ids:
            break
        num_cached_blocks += 1
        if block_id in self.used_block_ids:
            num_new_blocks -= 1
    if len(self.free_block_ids) < num_new_blocks:
        return -1
    return num_cached_blocks
```

### 逐步解读

**为什么不检查最后一块？**

```python
for i in range(seq.num_blocks - 1):
```

循环到 `num_blocks - 1`，**不检查最后一块**。因为**最后一块通常没满**
（比如 600 token → 最后一块只有 88 个，可能还会增长），
不满的块内容不稳定，不能作为缓存复用。只有**填满的块**才进哈希表。

**哈希链累积**

```python
h = self.compute_hash(token_ids, h)
```
每轮把上一个块的哈希 `h` 传进去，形成链。初始 `h = -1`（没有前缀）。

**查表并校验**

```python
block_id = self.hash_to_block_id.get(h, -1)
if block_id == -1 or self.blocks[block_id].token_ids != token_ids:
    break
```
- `-1`：没找到 → 停止（前缀到此为止）
- `token_ids` 不相等：哈希碰撞 → 停止，绝不误用

**命中计数与新块计数**

```python
num_cached_blocks += 1
if block_id in self.used_block_ids:
    num_new_blocks -= 1
```

- 命中一块，`num_cached_blocks++`
- 如果这块**正在被使用**（`used`），那么复用它是"借来的"，
  不需要新分配 → `num_new_blocks--`
- 如果这块**不在 used**（已被释放到 free 池），它本来就是空闲的，
  不需要减 `num_new_blocks`（因为 free 里已经算了它）

**显存检查**

```python
if len(self.free_block_ids) < num_new_blocks:
    return -1
return num_cached_blocks
```

需要的新块数超过空闲数 → 返回 `-1`（调度器会跳过这个序列）。
否则返回能复用的前缀块数。

### 这个函数的双重职责

⚠️ **注意它不只是"探测"，还返回了调度决策的依据**：
- 返回值 ≥ 0：能分配，且告诉你免费复用了几块
- 返回值 = -1：不能分配（显存不够）

它**没有副作用**（不修改任何状态），可以安全地"预演"。
真正分配是下一步 `allocate`。

### 一个具体例子

```
block_size = 256
序列C: prompt = [你好]×600, 即 3 块

库里已有：
  hash_to_block_id[h0] = 块3   tokens=[你好]×256
  hash_to_block_id[h1] = 块7   tokens=[你好]×256
  其中 块3 在 used 集合，块7 已释放

can_allocate(C):
  i=0: h0' = hash(块0, -1) = h0 → 块3, token 匹配 ✓
       num_cached_blocks=1, 块3 在 used → num_new_blocks = 3-1 = 2
  i=1: h1' = hash(块1, h0) = h1 → 块7, token 匹配 ✓
       num_cached_blocks=2, 块7 不在 used → num_new_blocks 不变 = 2
  (循环范围是 range(2) → i=0,1，最后一块不查)

  free_block_ids 有 5 个 ≥ 2 ✓
  return 2   ← 能复用 2 块！
```

## 4.6 `allocate()`：真正分配

`nanovllm/engine/block_manager.py:75`：

```python
def allocate(self, seq: Sequence, num_cached_blocks: int):
    assert not seq.block_table
    h = -1
    for i in range(num_cached_blocks):
        token_ids = seq.block(i)
        h = self.compute_hash(token_ids, h)
        block_id = self.hash_to_block_id[h]
        block = self.blocks[block_id]
        if block_id in self.used_block_ids:
            block.ref_count += 1
        else:
            block.ref_count = 1
            self.free_block_ids.remove(block_id)
            self.used_block_ids.add(block_id)
        seq.block_table.append(block_id)
    for i in range(num_cached_blocks, seq.num_blocks):
        seq.block_table.append(self._allocate_block())
    seq.num_cached_tokens = num_cached_blocks * self.block_size
```

### 两段式分配

**第一段：复用缓存的前缀块**（`0` 到 `num_cached_blocks-1`）

```python
if block_id in self.used_block_ids:
    block.ref_count += 1           # 别人正在用 → 加引用
else:
    block.ref_count = 1            # 空闲 → 占为己有
    self.free_block_ids.remove(block_id)
    self.used_block_ids.add(block_id)
seq.block_table.append(block_id)
```

分两种情况：
- **块正在被使用**（used 集合里有）：纯共享，`ref_count++` 即可
- **块在空闲池**（曾经被用过、已释放，但哈希记录还在）：
  从 free 池挪到 used，`ref_count = 1`

这第二种情况就是**前缀缓存"复活"已释放块**的魔法。

⚠️ 用 `self.free_block_ids.remove(block_id)`（O(n)）而不是 popleft，
因为块可能不在队首，找它得从头扫。这是个轻微的性能妥协
（vLLM 用了更高效的 LRU 结构）。

**第二段：分配全新块**（`num_cached_blocks` 到末尾）

```python
for i in range(num_cached_blocks, seq.num_blocks):
    seq.block_table.append(self._allocate_block())
```
需要多少给多少，从 free 池 popleft。

**记录已缓存 token 数**

```python
seq.num_cached_tokens = num_cached_blocks * self.block_size
```
让调度器知道"前缀已经就绪"，只需算后面的部分。

### `ref_count` 图例

```
分配前：
  free = [2, 4, 5, 6, 7]  used = {0, 3}
  hash_to_block_id = {h0: 3}     (块3 命中)
  Block[3].ref_count = 1

序列C allocate(num_cached_blocks=1)：
  块3 在 used → ref_count = 2
  C.block_table = [3, 2, 4]      ← 复用块3 + 新分配块2、块4

分配后：
  free = [5, 6, 7]  used = {0, 2, 3, 4}
  Block[3].ref_count = 2   ← C 和原主人共享
  Block[2].ref_count = 1
  Block[4].ref_count = 1
```

## 4.7 `_allocate_block` 与 `_deallocate_block`

`block_manager.py:43`：

```python
def _allocate_block(self) -> int:
    block_id = self.free_block_ids.popleft()
    block = self.blocks[block_id]
    assert block.ref_count == 0
    if block.hash != -1 and self.hash_to_block_id.get(block.hash) == block_id:
        del self.hash_to_block_id[block.hash]
    block.reset()
    self.used_block_ids.add(block_id)
    return block_id
```

要点：
- `assert ref_count == 0`：只有没人用的块能进 free 池
- **清理哈希记录**：这个块马上要被覆盖，它的旧哈希失效，
  从 `hash_to_block_id` 里删掉（防止别的前缀错误命中）
- `reset()` 为新主人初始化

`block_manager.py:53`：

```python
def _deallocate_block(self, block_id: int):
    assert self.blocks[block_id].ref_count == 0
    self.used_block_ids.remove(block_id)
    self.free_block_ids.append(block_id)
```

释放只是把它放回 free 池。**注意：不删哈希记录！**
这就是为什么被释放的块还能被前缀缓存"复活"（如 4.6 所述）。

## 4.8 `deallocate()`：序列结束时释放

`block_manager.py:94`：

```python
def deallocate(self, seq: Sequence):
    for block_id in reversed(seq.block_table):
        block = self.blocks[block_id]
        block.ref_count -= 1
        if block.ref_count == 0:
            self._deallocate_block(block_id)
    seq.num_cached_tokens = 0
    seq.block_table.clear()
```

- **逆序遍历**：从最后一块开始（虽然顺序其实无所谓，
  但逆序符合"释放栈"的直觉）
- 每块 `ref_count--`，减到 0 才真正放回 free 池
- 清空序列的块表

⚠️ `seq.num_cached_tokens = 0`：序列的"已缓存"归零。
因为它的块已经没了。这也是为什么被抢占的序列要重新 prefill。

## 4.9 `can_append` / `may_append`：decode 时的扩块

decode 每步生成 1 个 token，需要往 KV Cache 追加一个位置。
什么时候需要新块？当**当前块装满了**的时候。

### 关键：`len(seq) % block_size == 1` 是什么意思

`block_manager.py:103`：

```python
def can_append(self, seq: Sequence) -> bool:
    return len(self.free_block_ids) >= (len(seq) % self.block_size == 1)

def may_append(self, seq: Sequence):
    if len(seq) % self.block_size == 1:
        seq.block_table.append(self._allocate_block())
```

`len(seq) = num_tokens`。`num_tokens % block_size == 1` 表示：
**当前 token 是某个块的第 1 个**。

### 为什么？

先搞清楚"新增 token 是否开启新块"的判据。
假设 block_size = 256，已有 256 个 token（正好 1 整块）：

```
num_tokens = 256  →  num_tokens % 256 = 0   → 块满，下个 token 需要新块吗？
```

- `num_tokens = 256`：块0满了，第 257 个 token 需要块1。
  此时准备写第 257 个 token 前，`len(seq) = 256`，`256 % 256 = 0`。**没触发！**

再看：
- `num_tokens = 257`：已经写了第 257 个 token，块1有了 1 个 token。
  下一步准备写第 258 个，`len(seq) = 257`，`257 % 256 = 1` ✓

嗯？这里时序要对齐。关键：**`may_append` 在采样后、追加 token 前调用**吗？
不，看调度器顺序：

```python
seq.num_scheduled_tokens = 1
seq.is_prefill = False
self.block_manager.may_append(seq)     # ← 在这里调用
scheduled_seqs.append(seq)
```

然后 `model_runner.run` 写 KV，然后 `postprocess` 里 `append_token`。

所以 `may_append` 时 `len(seq)` **是上一步结束后的 num_tokens**（还没 append 新 token）。

**正确理解**：这一步要给**第 `len(seq)+1` 个** token 写 KV。
如果 `len(seq)+1` 是新块的第 1 个位置，就需要新块。
`len(seq)+1` 是块的第 1 个位置 ⟺ `(len(seq)+1) % block_size == 1`
⟺ `len(seq) % block_size == 0`...

等等，让我用具体数字确认。设 block_size=4（方便看）：

| num_tokens(len) | 这一步要写的 token 位置 | 属于哪块 | 需要新块？ | `len%4==1`? |
|---|---|---|---|---|
| 4 | 第5个 | 块1 | 是 | 4%4=0, 否 ✗ |
| 5 | 第6个 | 块1 | 否 | 5%4=1, 是 ✗ |
| 6 | 第7个 | 块1 | 否 | 6%4=2, 否 |
| 7 | 第8个 | 块1 | 否 | 7%4=3, 否 |
| 8 | 第9个 | 块2 | 是 | 8%4=0, 否 ✗ |

对不上！说明我对时序的假设需要修正。

### 修正：`len(seq)` 在 prefill 后的初始值

关键在于：**prefill 结束时，序列的第一个生成 token 已经可能已经占位**。
再想一遍完整的：
- prefill 处理 N 个 prompt token，出第 1 个生成 token（放进 token_ids）
- 此时 `num_tokens = N+1`

如果 N = 256（正好 1 块），则 `num_tokens = 257`，`257 % 256 = 1` ✓。
下一步 `may_append` 时 `len=257`，触发 → 分配块1。
第 258 个 token（第 2 个生成 token）会写入块1的第 2 个位置。

等一下，第 257 个 token（第 1 个生成 token）写哪了？它在 prefill 的 KV 写入里，
但 prefill 时 `slot_mapping` 是给 prompt token 用的，生成 token 的 KV
要等**下一次 decode** 才写（因为它的 K/V 需要前向才算得出来）。

让我重新理清——**生成 token 的 KV 是在生成它的那一步写入的**。
prefill 算 N 个 prompt token → 输出第 1 个生成 token t_{N}。
t_N 的 KV 要到**下一次 forward**（即以 t_N 为输入的 decode）才计算并写入。

所以：
- prefill 后：`num_tokens = N+1`（prompt N 个 + 生成 1 个），KV 里有 N 个位置
- 第一次 decode：输入 t_N，计算它的 K/V，写入位置 N（0-indexed），输出 t_{N+1}
  - 写之前 `len(seq) = N+1`
  - 如果 N+1 ≡ 1 (mod 256)，即 N ≡ 0 (mod 256)，需要新块

若 N = 256：`len = 257`，`257 % 256 = 1` ✓ 触发新块。写入位置 256（块1的第1个）。✓

正好对上！所以 `len(seq) % block_size == 1` 的直觉是：
**"num_tokens 刚好比整块多 1"时，要给第 (整块数+1) 个新块开张**。
因为 KV 位置数 = num_tokens - 1，当 KV 位置数是 block_size 的倍数时，需要新块。
`num_tokens - 1 ≡ 0 (mod block_size)` ⟺ `num_tokens ≡ 1 (mod block_size)`。

即：**KV 缓存里已经有整块数目的位置，下一个要写在下一块的开头**。

### `can_append` 的布尔运算技巧

```python
return len(self.free_block_ids) >= (len(seq) % self.block_size == 1)
```

Python 里 `True == 1`、`False == 0`，所以：
- 需要新块时（True）：检查 free ≥ 1
- 不需要时（False）：检查 free ≥ 0（永远成立）

一行搞定"需要时才检查"。

## 4.10 `hash_blocks()`：登记填满的块

`block_manager.py:110`：

```python
def hash_blocks(self, seq: Sequence):
    start = seq.num_cached_tokens // self.block_size
    end = (seq.num_cached_tokens + seq.num_scheduled_tokens) // self.block_size
    if start == end: return
    h = self.blocks[seq.block_table[start - 1]].hash if start > 0 else -1
    for i in range(start, end):
        block = self.blocks[seq.block_table[i]]
        token_ids = seq.block(i)
        h = self.compute_hash(token_ids, h)
        block.update(h, token_ids)
        self.hash_to_block_id[h] = block.block_id
```

在 `postprocess` 里、更新计数**之前**调用。

### 计算新填满的块范围

```python
start = seq.num_cached_tokens // self.block_size
end = (seq.num_cached_tokens + seq.num_scheduled_tokens) // self.block_size
```

`start`：之前已完整填满的块数（不含）
`end`：本步之后完整填满的块数

例：block_size=256，`num_cached_tokens=256`（1块已满），
本步处理 `num_scheduled_tokens=512` → `start=1, end=3`，
说明本步让块1、块2 填满了。**整除法自动实现了"只登记填满的块"**。

### 接续哈希链

```python
h = self.blocks[seq.block_table[start - 1]].hash if start > 0 else -1
```
从上一个块的 hash 接着算（保持链的连续性）。
`start == 0` 说明从头开始，用 -1。

### 登记

```python
for i in range(start, end):
    block = self.blocks[seq.block_table[i]]
    token_ids = seq.block(i)
    h = self.compute_hash(token_ids, h)
    block.update(h, token_ids)
    self.hash_to_block_id[h] = block.block_id
```

计算哈希、写进块对象、登记到全局哈希表。**这就是"存缓存"的动作。**

## 4.11 完整实例：前缀缓存命中

### 场景

三个请求**共享同一个系统提示词**（system prompt），长度 512（2 块）：

```
系统提示词 S = "You are a helpful assistant..." (512 token)

请求1: S + "问题A"
请求2: S + "问题B"
请求3: S + "问题C"
```

### 请求1 处理

`can_allocate`：库里没 S → 返回 0。`allocate` 分配全新的块。

假设分配块 0、1、2（S 占块0、1，问题A 占块2）。

算完后 `hash_blocks` 登记块0、块1（都满了）：

```
Block[0]: hash=h0, token_ids=S[0:256]
Block[1]: hash=h1, token_ids=S[256:512]
hash_to_block_id = {h0: 0, h1: 1}
```

### 请求2 处理

`can_allocate`：
```
i=0: h0'=hash(S[0:256],-1)=h0 → Block[0], token匹配 ✓
     num_cached_blocks=1, 块0在used → num_new_blocks--
i=1: h1'=hash(S[256:512],h0)=h1 → Block[1], token匹配 ✓
     num_cached_blocks=2, 块1在used → num_new_blocks--
return 2
```

**2 块前缀命中！** `allocate` 时块0、块1 的 `ref_count` 变成 2：

```
Block[0].ref_count = 2   ← 请求1、2 共享
Block[1].ref_count = 2
请求2.block_table = [0, 1, 新块]
```

**省下了 512 个 token 的 prefill 计算！** 这是 Prefix Caching 的威力。

### 请求3 处理

同理，再命中 2 块，`ref_count` 变成 3。

### 请求1 结束

`deallocate`：块0→2、块1→2、块2→0（释放）。
块0、块1 还在被请求2、3 用，不释放。

### 请求2、3 也结束

块0、块1 的 `ref_count` 归零 → 放回 free 池。
**但 `hash_to_block_id` 里的记录还在！**

### 新请求 4 来了，也带 S

```
can_allocate: 命中 h0、h1 → 块0、块1 虽然不在 used，
              但哈希记录还在 → num_cached_blocks = 2
allocate: 把块0、块1 从 free 池"复活"，ref_count = 1
```

**前缀缓存持续生效，即使中间没人用！**

### 缓存失效的时机

块被**重新分配给一个不同内容**时：
- `_allocate_block` 里 `del self.hash_to_block_id[block.hash]`
- 旧的哈希记录被清除，S 的缓存失效

所以前缀缓存是"软缓存"——**谁先用谁占，用满就挤掉**。

## 4.12 显存全生命周期图

```
                    add_request
                        │
                        ▼
              ┌──── can_allocate ────┐
              │   探测前缀命中数      │
              │   检查 free 够不够    │
              └──────────┬───────────┘
                 -1      │      ≥0
         ┌───────────────┴───────────────┐
         │                               │
         ▼                               ▼
    （等显存）                    allocate
    这步跳过                 ┌─ 命中块：ref_count++ 或复活
                             └─ 新块：从 free 池取
                                    │
                                    ▼
                          ┌─── running ───┐
                          │  decode 每步   │
                          │  may_append    │  ← 块满时扩块
                          │  hash_blocks   │  ← 登记新满的块
                          └───────┬────────┘
                                  │
                 ┌────────────────┴────────────────┐
                 │                                 │
            正常结束                          被 preempt
                 │                                 │
                 ▼                                 ▼
          deallocate                       deallocate
          ref_count--                      全部减引用
          归零则回 free 池                   seq 回 waiting
          (哈希记录保留!)                    (重算 KV)
```

## 4.13 本章小结

- PagedAttention 把 KV Cache 切成 256-token 的块，用块表管理，消除碎片。
- `Block` 有 `ref_count`（共享计数）、`hash`（缓存键）、`token_ids`（防碰撞）。
- **哈希链**：每个块的哈希包含前一块的哈希，保证只有完整前缀匹配才复用。
- 只登记**填满的块**（除法自动实现）。
- `can_allocate` 探测前缀命中数 + 显存够不够，无副作用。
- `allocate` 复用命中块（`ref_count++` 或从 free 池复活）、分配新块。
- 释放的块**保留哈希记录**，让前缀缓存能"复活"，直到块被重新分配。
- 被抢占/结束的序列释放所有块，`num_cached_tokens` 归零。
- `can_append`/`may_append` 处理 decode 时的扩块，判据是 `len(seq) % block_size == 1`。

**下一章**：[ModelRunner](05-model-runner.md)——把逻辑序列翻译成 GPU 张量。
