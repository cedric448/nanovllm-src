# 导读：Nano-vLLM 架构总览

## 0.1 一句话概括

Nano-vLLM 是一个**离线批量推理引擎**：你给它一堆 prompt，它高效地（利用 GPU 的每一分显存和算力）
把这些 prompt 的续写全部算出来。

"高效"体现在三个地方，这也是全书的三大主线：

1. **调度高效** —— 不同长度的请求可以同时跑，不会互相干等（第 3 章）
2. **显存高效** —— KV Cache 按块管理，相同前缀自动复用（第 4 章）
3. **算力高效** —— 分页 attention、flash-attn、CUDA Graph、张量并行（第 5、6、7 章）

## 0.2 为什么需要"推理引擎"而不是直接调 HuggingFace？

用 `model.generate()` 也能出结果，那 vLLM/Nano-vLLM 到底解决了什么问题？先看两个朴素做法的痛点。

### 痛点一：显存浪费（静态分配）

朴素做法：给每个请求预留一个固定长度的 KV Cache，比如都按 `max_model_len=4096` 预留。

假设 3 个请求，实际长度分别是 10、20、4096：

```
朴素方案（按最大长度预留）：
请求A [████░░░░░░░░░░░░░░░░░░░░░░░░░░]  只用 0.2%，白占 99.8%
请求B [██████░░░░░░░░░░░░░░░░░░░░░░░░]  只用 0.5%
请求C [██████████████████████████████]  满了

显存利用率 ≈ (10+20+4096) / (4096×3) ≈ 33%
```

Nano-vLLM 的答案：**分页**（第 4 章）。KV Cache 切成 256-token 一块，按需分配：

```
分页方案（按块分配）：
请求A [A0░░]                          只占 1 块
请求B [B0░░]                          只占 1 块
请求C [C0][C1][C2]...[C15]            占 16 块
剩余的空块可以给别人用 → 利用率接近 100%
```

### 痛点二：调度僵化（串行 / 填充）

朴素做法有两种极端，都不好：

```
做法1：一次只跑一个请求（串行）
[请求A: prefill+decode 全跑完] → [请求B...] → [请求C...]
问题：GPU 大部分时间在等一个请求，batch 太小，算力浪费

做法2：所有请求攒齐一起跑（静态 batch）
等 A、B、C 全部到齐才开始
问题：A 很短但 C 很长，A 早算完了还得等 C；新请求进不来
```

Nano-vLLM 的答案：**连续批处理（Continuous Batching）**（第 3 章）。
每一步（step）都重新决定"这一步跑哪些请求"，跑完的立刻退出、新来的立刻加入：

```
Step 1: [A prefill][B prefill][C prefill]
Step 2: [A decode][B decode][C decode]
Step 3: [A decode][B decode][C decode]   ← A 生成完了，这一步后退出
Step 4: [B decode][C decode][D prefill]  ← D 新加入，和 B/C 一起跑
...
```

## 0.3 整体架构

代码极简，只有三个模块目录：

```
nanovllm/
├── llm.py                     # 对外 API（就 5 行，一个空继承）
├── config.py                  # 引擎配置
├── sampling_params.py         # 采样参数
│
├── engine/                    # 【调度层 + 执行层】
│   ├── llm_engine.py          #   引擎主循环（心跳）
│   ├── scheduler.py           #   调度器（决定每一步跑谁）
│   ├── block_manager.py       #   分页显存管理（PagedAttention 的核心）
│   ├── sequence.py            #   一条请求的状态载体
│   └── model_runner.py        #   GPU 执行器（组装张量、跑模型）
│
├── layers/                    # 【算子层】
│   ├── attention.py           #   Attention + Triton KV 写入kernel
│   ├── linear.py              #   张量并行的四种线性层
│   ├── embed_head.py          #   词表并行的 Embedding / LM Head
│   ├── rotary_embedding.py    #   RoPE 旋转位置编码
│   ├── layernorm.py           #   RMSNorm
│   ├── activation.py          #   SwiGLU 激活
│   └── sampler.py             #   采样
│
├── models/
│   └── qwen3.py               #   Qwen3 模型结构定义
│
└── utils/
    ├── context.py             #   全局上下文（贯穿各层的信息传递）
    └── loader.py              #   从 safetensors 加载权重
```

## 0.4 数据流全景图

这是全书最重要的一张图。**建议对照它读每一章**。

```
用户调用
  │
  │  llm.generate(prompts, sampling_params)                    [第 1 章]
  ▼
┌─────────────────────────────────────────────────────────────┐
│ LLMEngine.generate()                                        │
│   ├─ add_request()  → 每个 prompt 变成一个 Sequence          │  [第 2 章]
│   │                    塞进 scheduler.waiting 队列            │
│   └─ while not is_finished():                               │
│        └─ step()          ← 引擎的"心跳"，每跳一次生成一批 token
└─────────────────────────────────────────────────────────────┘
  │
  │  step() 三步走：
  ▼
① Scheduler.schedule()                                          [第 3 章]
  ├─ 有 waiting 就做 prefill：按 token 预算挑选序列
  ├─ 否则做 decode：每个 running 序列分配 1 个 token
  └─ 返回 (选中的序列列表, 是不是prefill)
  │
  │  同时，BlockManager 被调用：                                [第 4 章]
  │  can_allocate() → allocate()（分配 KV 块，尝试前缀复用）
  ▼
② ModelRunner.run(seqs, is_prefill)                             [第 5 章]
  ├─ prepare_prefill/decode()：把 Sequence 翻译成 GPU 张量
  │    （input_ids, positions, slot_mapping, block_tables...）
  ├─ 写进全局 Context                                          [utils/context.py]
  ├─ model(input_ids, positions)   ← Qwen3 前向                 [第 7 章]
  │    └─ 每层 Attention:                                      [第 6 章]
  │         ├─ store_kvcache()：把本步的 K/V 写进分页 cache
  │         └─ flash_attn：读 cache 算 attention
  └─ sampler(logits)：采样出下一个 token
  ▼
③ Scheduler.postprocess()                                       [第 3 章]
  ├─ 把新 token 写回 Sequence
  ├─ 检查是否遇到 EOS 或达到 max_tokens
  └─ 完成的序列释放 KV 块
  │
  │  回到 while 循环，直到所有序列完成
  ▼
输出：解码成文本的 list[dict]
```

## 0.5 三个"反直觉"的设计决策

读代码时容易困惑的地方，先在这里给出解释。

### 决策一：为什么 prefill 和 decode 要分开调度？

因为它们的**计算形态完全不同**：

| | Prefill | Decode |
|---|---|---|
| 输入 | 整段 prompt（N 个 token） | 只有 1 个 token |
| 矩阵形状 | 方阵（N×N） | 向量-矩阵（1×N） |
| 瓶颈 | 算力（GPU 计算单元打满） | 显存带宽（搬运 KV 到计算单元） |
| 适合 batch | 少量长序列 | 大量短序列 |

把长 prompt 的 prefill 和一堆 decode 混在一起跑，prefill 会算很久，
让所有 decode 请求卡住（TTFT 爆炸）。所以 Nano-vLLM 优先把 prefill 做完，
再用 decode 把 GPU 填满。

### 决策二：为什么需要 Chunked Prefill？

一个 4096 token 的 prompt，prefill 要一次性算完 4096 个 token。
如果 GPU 一次最多处理 16384 个 token（`max_num_batched_tokens`），
那一个超长 prompt 就霸占了全部预算，其他请求都得等。

Chunked Prefill 把它切成几块，分几个 step 喂：

```
不分块：Step1 = [4096 token 的大 prefill]，其他请求饿死
分块：  Step1 = [1024 token chunk] + 其他请求的 decode
        Step2 = [1024 token chunk] + ...
        ...
```

⚠️ 注意：Nano-vLLM 有个简化——**只有第一个序列允许被分块**（第 3 章细讲）。

### 决策三：为什么 KV Cache 要写成巨大的单个 tensor？

`allocate_kv_cache()`（第 5 章）里，KV Cache 是一个形状为
`[2, num_layers, num_blocks, block_size, num_kv_heads, head_dim]` 的大 tensor，
然后切片挂到每层 attention 上。

这样做的好处：
1. **一次分配，永不重分配** —— 避免 CUDA 内存碎片，这是推理服务的稳定性关键；
2. **块号 → 物理地址** 就是简单的乘法，无需查表（第 6 章的 Triton kernel 直接算地址）；
3. 前缀复用、抢占都能在这个平面地址空间里自由腾挪。

## 0.6 一张表看懂"每一步都在动什么"

以 3 个请求为例，看 Sequence 和 BlockManager 的状态如何随 step 演化。
（这是第 8 章端到端走查的浓缩版，先建立直觉）

| Step | 做什么 | waiting | running | 发生的事 |
|------|--------|---------|---------|----------|
| 0 | add_request × 3 | A,B,C | — | 3 个 Sequence 入队 |
| 1 | prefill | — | A,B,C | 分配 KV 块，算完 prompt，各出 1 个 token |
| 2 | decode | — | A,B,C | 各写 1 个 KV，各出 1 个 token |
| 3 | decode | — | A,B,C | A 遇到 EOS → 结束，释放它的块 |
| 4 | decode | — | B,C | 只有 B、C 在跑 |
| 5 | decode | — | — | B、C 都结束，`is_finished()` 为真，退出循环 |

## 0.7 关键参数一览（config.py）

理解这些数字，才能理解调度器的决策：

| 参数 | 默认值 | 含义 | 影响 |
|------|--------|------|------|
| `max_num_batched_tokens` | 16384 | 单次前向最多处理多少 token | 越大吞吐越高，但延迟越大 |
| `max_num_seqs` | 512 | 同时活跃的请求上限 | 越大并发越高，但显存压力越大 |
| `max_model_len` | 4096 | 单条序列最大长度 | 被 `min(配置, 模型上限)` 截断 |
| `gpu_memory_utilization` | 0.9 | KV Cache 占用显存比例 | 决定能分配多少个 block |
| `kvcache_block_size` | 256 | 一个 KV 块装多少 token | 必须是 256 的倍数 |
| `tensor_parallel_size` | 1 | 用几张卡 | >1 会 spawn 子进程 |

## 0.8 开始阅读

准备好了吗？从 **[第 1 章：入口与 API](01-entry-api.md)** 开始。

如果你想先看一个完整的例子建立直觉，可以直接跳到
**[第 8 章：端到端走查](08-walkthrough.md)**，再回头按顺序读。
