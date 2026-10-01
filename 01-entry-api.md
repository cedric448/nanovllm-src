# 第 1 章 入口与 API

> 本章目标：搞清楚"一行 `llm.generate()` 究竟触发了什么"，
> 以及整个引擎最外层的几个配置类。

## 1.1 对外只有两个东西

Nano-vLLM 刻意把 API 面收得极窄。用户需要知道的只有两个类：

```python
from nanovllm import LLM, SamplingParams

llm = LLM("/path/to/Qwen3-0.6B", enforce_eager=True, tensor_parallel_size=1)
sampling_params = SamplingParams(temperature=0.6, max_tokens=256)
outputs = llm.generate(["Hello, Nano-vLLM."], sampling_params)
outputs[0]["text"]  # 生成的文本
```

看导入源头 `nanovllm/__init__.py`：

```python
from nanovllm.llm import LLM
from nanovllm.sampling_params import SamplingParams
```

而 `nanovllm/llm.py` 全文只有 5 行：

```python
from nanovllm.engine.llm_engine import LLMEngine

class LLM(LLMEngine):
    pass
```

**为什么可以这么空？** 因为 `LLM` 只是 `LLMEngine` 的一个别名式继承。
真正干活的全在 `LLMEngine` 里。这种写法在 vLLM 里也常见——
保留一个"门面类"，方便以后替换实现或加包装，而不影响用户代码。

## 1.2 `SamplingParams`：采样参数

`nanovllm/sampling_params.py`：

```python
from dataclasses import dataclass

@dataclass(slots=True)
class SamplingParams:
    temperature: float = 1.0
    max_tokens: int = 64
    ignore_eos: bool = False

    def __post_init__(self):
        assert self.temperature > 1e-10, "greedy sampling is not permitted"
```

三个字段，含义直白：

| 字段 | 含义 | 说明 |
|------|------|------|
| `temperature` | 采样温度 | 越大越随机，趋近 0 越确定 |
| `max_tokens` | 最多生成多少 token | 达到就停 |
| `ignore_eos` | 是否忽略结束符 | `True` 时即使遇到 EOS 也继续生成 |

### 一个反直觉的约束：不允许贪心

注意那句断言：`temperature` 必须严格大于 `1e-10`，否则直接报错。

**为什么要禁止贪心（temperature=0）？**

因为贪心采样需要 `argmax`，而这里用的是**多项分布采样**（第 7 章 sampler）。
采样实现是 `probs / exponential(1) 取 argmax` 的技巧（Gumbel-max 变体），
当 `temperature → 0` 时除以 0 会变成 `inf`，数值不稳定。

这是 Nano-vLLM 的一个简化取舍：**牺牲贪心，换取采样实现极其简洁**。
如果确实需要确定性输出，可以设一个极小温度（如 0.01）。

### `slots=True` 是什么？

`@dataclass(slots=True)` 让实例用 `__slots__` 存储字段，
每个实例不再带一个 `__dict__`，省内存、访问更快。

因为推理引擎会创建**成千上万个 Sequence**（每条请求一个），
每个都省下几十字节的 dict 开销就很可观。

## 1.3 `Config`：引擎全局配置

`nanovllm/config.py`：

```python
import os
from dataclasses import dataclass
from transformers import AutoConfig

@dataclass(slots=True)
class Config:
    model: str
    max_num_batched_tokens: int = 16384
    max_num_seqs: int = 512
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.9
    tensor_parallel_size: int = 1
    enforce_eager: bool = False
    hf_config: AutoConfig | None = None
    eos: int = -1
    kvcache_block_size: int = 256
    num_kvcache_blocks: int = -1

    def __post_init__(self):
        assert os.path.isdir(self.model)
        assert self.kvcache_block_size % 256 == 0
        assert 1 <= self.tensor_parallel_size <= 8
        self.hf_config = AutoConfig.from_pretrained(self.model)
        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)
```

### 逐条解读

**`model`**：模型权重目录（本地路径）。断言要求它是个目录。

**`max_num_batched_tokens = 16384`**：*调度预算*。
一次 `step()` 里，所有序列加起来最多处理这么多 token。
这是调度的硬约束，第 3 章会反复用到。

**`max_num_seqs = 512`**：*并发上限*。同时活跃的序列不超过 512 条。

**`max_model_len = 4096`**：单条序列最大长度。注意最后一行：

```python
self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)
```

用模型的真实上限兜底。假如模型只能处理 32768 个位置，你配了 4096，就用 4096；
如果你配了 100000 但模型只支持 32768，会被自动砍到 32768，防止越界。

**`gpu_memory_utilization = 0.9`**：KV Cache 可以用掉 90% 的显存。
剩下的留给模型权重、激活值、CUDA context。第 5 章用它反推 block 数量。

**`tensor_parallel_size`**：用几张卡。断言限制在 1~8，因为 NCCL 通信端口、
shared memory 等在这个极简实现里是按小规模设计的。

**`enforce_eager = False`**：是否禁用 CUDA Graph。
`False` 会用 CUDA Graph 加速 decode；调试时设 `True` 更容易定位问题。
`example.py` 里就设了 `True`。

**`kvcache_block_size = 256`**：一个 KV 块装 256 个 token。
断言要求是 256 的倍数——这是因为 flash-attn 的 block_table 有对齐要求。

**`num_kvcache_blocks = -1`**：初始占位符，运行时由 `allocate_kv_cache()` 填真实值。

**`eos = -1`**：初始占位符，`LLMEngine.__init__` 里从 tokenizer 读取真实值。

### 这些 `-1` 占位符的设计意图

`Config` 在构造时还不知道实际显存和 tokenizer 信息，
所以先给哨兵值，等引擎初始化时再回填。
第 5 章会看到 `config.num_kvcache_blocks = ...` 的赋值，
第 1.4 节会看到 `config.eos = self.tokenizer.eos_token_id`。

## 1.4 核心：`LLMEngine.__init__` 做了什么

`nanovllm/engine/llm_engine.py:17`：

```python
class LLMEngine:

    def __init__(self, model, **kwargs):
        config_fields = {field.name for field in fields(Config)}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        config = Config(model, **config_kwargs)
        Sequence.block_size = config.kvcache_block_size
        self.ps = []
        self.events = []
        ctx = mp.get_context("spawn")
        for i in range(1, config.tensor_parallel_size):
            event = ctx.Event()
            process = ctx.Process(target=ModelRunner, args=(config, i, event))
            process.start()
            self.ps.append(process)
            self.events.append(event)
        self.model_runner = ModelRunner(config, 0, self.events)
        self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)
        config.eos = self.tokenizer.eos_token_id
        self.scheduler = Scheduler(config)
        atexit.register(self.exit)
```

逐步拆解：

### (1) 从 kwargs 里挑出 Config 字段

```python
config_fields = {field.name for field in fields(Config)}
config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
config = Config(model, **config_kwargs)
```

用户 `LLM(path, enforce_eager=True, tensor_parallel_size=1)` 传的
`enforce_eager`、`tensor_parallel_size` 会被挑出来传给 `Config`。
这个"过滤"设计让 `LLM()` 的签名很宽松——传了不认识的参数会被静默忽略。

### (2) 把 block_size 传给 Sequence 类

```python
Sequence.block_size = config.kvcache_block_size
```

注意这是**类属性**赋值。`Sequence` 需要 block_size 来计算块数，
但每条序列都存一份太浪费，所以做成类级别的共享常量。

### (3) 启动张量并行的子进程

```python
ctx = mp.get_context("spawn")
for i in range(1, config.tensor_parallel_size):
    event = ctx.Event()
    process = ctx.Process(target=ModelRunner, args=(config, i, event))
    process.start()
    self.ps.append(process)
    self.events.append(event)
```

当 `tensor_parallel_size=1`（默认）时，这个循环**一次都不执行**，
所以你平时看不到任何子进程。

当 `tensor_parallel_size=2` 时，启动 rank=1 的子进程，它一启动就在
`ModelRunner.__init__` 末尾进入 `self.loop()` 死循环等指令（第 5 章细讲）。

用 `spawn` 而非 `fork`：CUDA 上下文不能跨 fork 安全共享，spawn 更干净。
代价是启动慢、需要能重新导入模块。

### (4) 主进程自己也是 rank 0

```python
self.model_runner = ModelRunner(config, 0, self.events)
```

主进程直接实例化 rank=0 的 ModelRunner（不新开进程）。
这样单卡时零额外开销，多卡时 rank0 是主控。

### (5) 补全 config 的两个占位符

```python
self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)
config.eos = self.tokenizer.eos_token_id
```

现在知道 EOS token id 了，回填到 config。scheduler 需要它来判断序列结束。

### (6) 创建调度器，注册退出钩子

```python
self.scheduler = Scheduler(config)
atexit.register(self.exit)
```

`atexit` 确保程序退出时调用 `exit()`：
- 通知所有 ModelRunner 退出循环
- 等待子进程 join
- 清理共享内存、CUDA Graph、NCCL

如果不注册，多卡场景下子进程可能变僵尸进程、共享内存泄漏。

### 初始化时序图

```
LLMEngine.__init__
  ├─ 构造 Config（读 HF 配置，回填 max_model_len）
  ├─ Sequence.block_size = 256
  ├─ spawn rank=1..N-1 的 ModelRunner  ──→ 子进程加载模型→loop()等待
  ├─ 构造 rank=0 的 ModelRunner（加载模型、分配 KV、捕获 CUDA Graph）
  ├─ 读 tokenizer，回填 config.eos
  ├─ 创建 Scheduler（创建 BlockManager）
  └─ atexit.register(exit)
```

⚠️ **初始化很慢**：要加载权重、warmup、分配 KV Cache、捕获多个 CUDA Graph。
这正是为什么引擎设计成"常驻服务"，一次初始化、多次生成。

## 1.5 核心：`generate()` 主循环

`nanovllm/engine/llm_engine.py:60`：

```python
def generate(
    self,
    prompts: list[str] | list[list[int]],
    sampling_params: SamplingParams | list[SamplingParams],
    use_tqdm: bool = True,
) -> list[str]:
    pbar = tqdm(total=len(prompts), desc="Generating", dynamic_ncols=True, disable=not use_tqdm)
    if not isinstance(sampling_params, list):
        sampling_params = [sampling_params] * len(prompts)
    for prompt, sp in zip(prompts, sampling_params):
        self.add_request(prompt, sp)
    outputs = {}
    prefill_throughput = decode_throughput = 0.
    while not self.is_finished():
        t = perf_counter()
        output, num_tokens = self.step()
        if num_tokens > 0:
            prefill_throughput = num_tokens / (perf_counter() - t)
        else:
            decode_throughput = -num_tokens / (perf_counter() - t)
        pbar.set_postfix({
            "Prefill": f"{int(prefill_throughput)}tok/s",
            "Decode": f"{int(decode_throughput)}tok/s",
        })
        for seq_id, token_ids in output:
            outputs[seq_id] = token_ids
            pbar.update(1)
    pbar.close()
    outputs = [outputs[seq_id] for seq_id in sorted(outputs.keys())]
    outputs = [{"text": self.tokenizer.decode(token_ids), "token_ids": token_ids} for token_ids in outputs]
    return outputs
```

### 分段解读

**(a) 参数广播**：如果只传一个 `SamplingParams`，广播给所有 prompt。
也支持每个 prompt 不同的采样参数（`bench.py` 就这么用）。

**(b) 全部入队**：

```python
for prompt, sp in zip(prompts, sampling_params):
    self.add_request(prompt, sp)
```
`add_request`（见 1.6）把 prompt 变成 `Sequence` 塞进 waiting 队列。
注意：**所有请求一次性全塞进去**，而不是一个个来。
调度器会自己决定每步跑多少（这正是 continuous batching 的入口）。

**(c) 主循环**：

```python
while not self.is_finished():
    output, num_tokens = self.step()
```

`is_finished()` 就是检查 `waiting` 和 `running` 都空了（第 3 章）。
每次 `step()` 产出一批 token。循环直到所有序列完成。

**(d) 吞吐统计的小技巧**：

```python
num_tokens = sum(seq.num_scheduled_tokens for seq in seqs) if is_prefill else -len(seqs)
```

> ⚠️ 这里是全书最"皮"的一行。`step()` 返回的 `num_tokens`
> 用**正负号来编码是 prefill 还是 decode**：
> - prefill：返回正数 = 本步处理的 token 数
> - decode：返回负数 = 本步解码的序列数
>
> 所以 `generate` 里 `if num_tokens > 0` 判断 prefill，
> `-num_tokens` 取回 decode 的真实数量。这是为了少传一个布尔值。

**(e) 输出排序**：

```python
outputs = [outputs[seq_id] for seq_id in sorted(outputs.keys())]
```

用 dict 按 `seq_id` 收集，最后按 id 排序还原成输入顺序。
因为序列是并发完成的，完成顺序和输入顺序不一致，必须重排。

### 一个 `generate()` 的完整时间线例子

假设 2 个 prompt，长度分别是 5 和 3：

```
generate(["你好", "世界"])
  │
  ├─ add_request("你好") → Sequence#0 (5 tokens) 入 waiting
  ├─ add_request("世界") → Sequence#1 (3 tokens) 入 waiting
  │
  ├─ step() #1  ├─ schedule → prefill: [Seq#0, Seq#1]
  │             ├─ model_runner.run([#0,#1], is_prefill=True)
  │             └─ postprocess → 各出 1 个 token
  │
  ├─ step() #2  ├─ schedule → decode: [Seq#0, Seq#1]
  │             └─ 各出 1 个 token
  │
  ├─ ... 直到某序列遇到 EOS 或到达 max_tokens
  │
  └─ is_finished() → True，退出
```

## 1.6 `add_request` 与 `step`

`llm_engine.py:43`：

```python
def add_request(self, prompt: str | list[int], sampling_params: SamplingParams):
    if isinstance(prompt, str):
        prompt = self.tokenizer.encode(prompt)
    seq = Sequence(prompt, sampling_params)
    self.scheduler.add(seq)
```

支持两种输入：
- 字符串 → 用 tokenizer 编码成 token id 列表
- 已经是 token id 列表 → 直接用（`bench.py` 用随机 token 测吞吐就是走这条路）

`llm_engine.py:49`：

```python
def step(self):
    seqs, is_prefill = self.scheduler.schedule()
    num_tokens = sum(seq.num_scheduled_tokens for seq in seqs) if is_prefill else -len(seqs)
    token_ids = self.model_runner.call("run", seqs, is_prefill)
    self.scheduler.postprocess(seqs, token_ids, is_prefill)
    outputs = [(seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished]
    return outputs, num_tokens
```

`step()` 是引擎的心跳，就四件事：

1. **`schedule()`** —— 决定这步跑谁（第 3 章）
2. **`call("run", ...)`** —— 让 GPU 执行（第 5 章）。
   用 `call` 而不是直接 `model_runner.run`，是因为多卡时 `call` 会先
   把指令广播给其他 rank。
3. **`postprocess()`** —— 把结果写回序列、判断结束（第 3 章）
4. **收集完成的输出** —— `seq.is_finished` 的序列产出结果

## 1.7 `exit()`：优雅退出

`llm_engine.py:37`：

```python
def exit(self):
    self.model_runner.call("exit")
    del self.model_runner
    for p in self.ps:
        p.join()
```

`call("exit")` 广播退出指令 → rank>0 的子进程从 `loop()` 跳出、退出。
`del self.model_runner` 触发 rank0 的 `__del__`/资源释放。
最后 `join()` 等所有子进程结束。

## 1.8 `example.py` 与 `bench.py` 导读

### `example.py`：最小可用示例

```python
path = os.path.expanduser("~/huggingface/Qwen3-0.6B/")
tokenizer = AutoTokenizer.from_pretrained(path)
llm = LLM(path, enforce_eager=True, tensor_parallel_size=1)

sampling_params = SamplingParams(temperature=0.6, max_tokens=256)
prompts = ["introduce yourself", "list all prime numbers within 100"]
prompts = [tokenizer.apply_chat_template(
    [{"role": "user", "content": prompt}],
    tokenize=False, add_generation_prompt=True) for prompt in prompts]

outputs = llm.generate(prompts, sampling_params)
```

两个要点：
1. **`enforce_eager=True`**：示例关掉了 CUDA Graph，方便调试
2. **`apply_chat_template`**：把对话格式化成 Qwen3 期望的 prompt。
   Nano-vLLM 本身不管 chat template，用户要自己套。

### `bench.py`：吞吐基准

```python
seed(0)
num_seqs = 256
max_input_len = 1024
max_ouput_len = 1024
llm = LLM(path, enforce_eager=False, max_model_len=4096)

prompt_token_ids = [[randint(0, 10000) for _ in range(randint(100, max_input_len))] for _ in range(num_seqs)]
sampling_params = [SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=randint(100, max_ouput_len)) for _ in range(num_seqs)]

llm.generate(["Benchmark: "], SamplingParams())   # warmup
t = time.time()
llm.generate(prompt_token_ids, sampling_params, use_tqdm=False)
t = (time.time() - t)
total_tokens = sum(sp.max_tokens for sp in sampling_params)
throughput = total_tokens / t
print(f"Total: {total_tokens}tok, Time: {t:.2f}s, Throughput: {throughput:.2f}tok/s")
```

要点：
- 用**随机 token id**（不走 tokenizer），纯粹测引擎吞吐
- `ignore_eos=True` 强制生成满 `max_tokens`，避免因提前结束影响测量
- 预热一次（`generate(["Benchmark: "], ...)`）确保 CUDA Graph、cache 就绪
- `use_tqdm=False` 关掉进度条，避免干扰计时

### 如何跑这个基准

```bash
pip install git+https://github.com/GeeeekExplorer/nano-vllm.git
huggingface-cli download --resume-download Qwen/Qwen3-0.6B --local-dir ~/huggingface/Qwen3-0.6B/
python example.py
python bench.py
```

## 1.9 本章小结

- 对外 API 只有 `LLM` + `SamplingParams`，`LLM` 是 `LLMEngine` 的薄壳。
- `Config` 集中所有引擎参数，用 `-1` 哨兵值 + 运行时回填解决"初始化时信息不全"。
- `LLMEngine.__init__` 的职责：建配置、spawn 子进程、加载模型、建调度器、注册清理钩子。
- `generate()` 是"全部入队 → while 循环 step → 收集输出"。
- `step()` 是引擎心跳：schedule → run → postprocess。
- `stop_condition`：遇到 EOS（除非 ignore_eos）或达到 max_tokens。

**下一章**：[Sequence 与采样参数](02-sequence-sampling.md)——
看看"一条请求"在引擎内部是如何被表示的。
