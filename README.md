# Nano-vLLM 源码精读

> 一本写给"想真正看懂大模型推理引擎内部原理"的读者的小书。
>
> 以 [GeeeekExplorer/nano-vllm](https://github.com/GeeeekExplorer/nano-vllm) 为对象，
> 逐章拆解一个约 1200 行的极简 vLLM 实现。每一章都包含 **原理**（为什么这么做）、
> **实现**（代码怎么写的）、**实例**（一个能算得清的数字例子）、**易错点**（坑在哪）。

---

## 这本书适合谁

- 会用 vLLM 但说不清它内部怎么调度、怎么管显存的工程师；
- 想动手写自己的推理引擎，但被 PagedAttention / Prefix Caching / CUDA Graph 这些名词劝退的人；
- 准备面试大模型推理岗位、需要"能画出数据流"的候选人。

阅读本书只需要：**会用 PyTorch**，知道 attention 是什么，听过 KV Cache 这个词。
不需要懂 CUDA、不需要读过 vLLM。

---

## 全书地图

代码分三层，本书也按这个脉络组织：

```
┌──────────────────────────────────────────────────────────────┐
│  用户层    LLM.generate(prompts, sampling_params)             │  ← 第 1 章
├──────────────────────────────────────────────────────────────┤
│  调度层    Scheduler  ──  谁这一步跑、跑多少 token             │  ← 第 3 章
│            BlockManager ── KV 放哪、能否复用前缀               │  ← 第 4 章
│            Sequence ───── 一条请求的完整状态载体               │  ← 第 2 章
├──────────────────────────────────────────────────────────────┤
│  执行层    ModelRunner ── batch 组装、KV 分配、CUDA Graph      │  ← 第 5 章
│            Attention ──── KV 读写、flash-attn 调用             │  ← 第 6 章
│            Qwen3/Linear ─ 模型结构、张量并行                   │  ← 第 7 章
└──────────────────────────────────────────────────────────────┘
```

## 目录

| 章节 | 文件 | 主题 | 核心问题 |
|------|------|------|----------|
| 导读 | [00-overview.md](00-overview.md) | 架构总览与数据流 | 这套系统到底在干什么？ |
| 第 1 章 | [01-entry-api.md](01-entry-api.md) | 入口与 API | 一行 `generate()` 背后发生了什么？ |
| 第 2 章 | [02-sequence-sampling.md](02-sequence-sampling.md) | Sequence 与采样参数 | 一条请求的状态如何表达？ |
| 第 3 章 | [03-scheduler.md](03-scheduler.md) | 调度器 | 每一步该跑哪些请求？ |
| 第 4 章 | [04-block-manager.md](04-block-manager.md) | 分页显存管理 | KV Cache 如何按块复用？ |
| 第 5 章 | [05-model-runner.md](05-model-runner.md) | GPU 执行器 | 逻辑序列如何变成 GPU 张量？ |
| 第 6 章 | [06-attention.md](06-attention.md) | Attention 与 KV 写入 | KV 怎么存、attention 怎么算？ |
| 第 7 章 | [07-model-tp.md](07-model-tp.md) | 模型与张量并行 | 权重如何切、多卡如何协作？ |
| 第 8 章 | [08-walkthrough.md](08-walkthrough.md) | 端到端走查 | 三个请求，一步步走完全程 |

## 推荐阅读路线

**第一次读（理解主线，约 2 小时）**
```
00 导读 → 01 入口 → 03 调度器 → 04 分页管理 → 08 端到端走查
```

**想搞懂性能优化**
```
05 ModelRunner（KV 分配 + CUDA Graph） → 06 Attention（flash-attn）
```

**想搞懂多卡**
```
07 模型与张量并行 → 05 ModelRunner 的进程通信部分
```

## 代码版本

- 仓库：`GeeeekExplorer/nano-vllm`
- 提交：`bb823b3`（chunked-prefill-refactor 合并后）
- 代码行数：约 1450 行 Python（含空行）

## 术语速查

| 术语 | 一句话解释 | 详见 |
|------|-----------|------|
| Prefill | 一次性处理整段 prompt，建立 KV Cache | 第 3、5、6 章 |
| Decode | 每次只生成 1 个 token | 第 3、5、6 章 |
| KV Cache | 缓存历史 token 的 Key/Value，避免重复计算 | 第 4、6 章 |
| PagedAttention | 把 KV Cache 切成固定大小的块来管理 | 第 4 章 |
| Prefix Caching | 相同前缀的请求复用同一批 KV 块 | 第 4 章 |
| TTFT | Time To First Token，首 token 延迟 | 第 3 章 |
| TPOT | Time Per Output Token，单 token 延迟 | 第 3 章 |
| Chunked Prefill | 长 prompt 分多次喂，避免阻塞其他请求 | 第 3 章 |
| Preemption | 显存不够时把请求踢出、释放 KV | 第 3 章 |
| CUDA Graph | 把一段 GPU 计算录制成图，回放省调度开销 | 第 5 章 |
| Tensor Parallel | 把权重切到多张卡上并行计算 | 第 7 章 |
| GQA | 多个 Query head 共享一组 KV head | 第 7 章 |
