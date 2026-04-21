---
layout: post
title: "vLLM - Revisit"
date: 2026-04-20
categories: [AI, Infrastructure]
tags: [LLM, Inference, Optimization]
description: A technical summary of PagedAttention, hardware nuances, and continuous batching in vLLM.
---

*Estimated read time: 12 minutes*

In the world of Large Language Model (LLM) inference, the primary bottleneck isn't just compute—it's memory management. Specifically, the management of the **Key-Value (KV) Cache**. vLLM has emerged as the industry standard by borrowing a classic concept from Operating Systems: **Virtual Memory** [^ref-vllm-2023].

## 1. The Core Innovation: PagedAttention
Traditional inference engines allocate KV cache in large, contiguous blocks. Because we don't know the output length in advance, systems "over-reserve" space, leading to **Internal Fragmentation** (wasted space within a sequence) and **External Fragmentation**.

**PagedAttention** solves this by partitioning the KV cache into small, fixed-size **Physical Blocks**. A **Block Table** maps these logical sequences to physical locations, allowing memory to be allocated on-demand. The following animation [^ref-vllm-blog] illustrates this mapping process.

<p align="center">
  <img src="/images/inference-2026-vllm/animation.gif" width="700">
  <br />
  <em>Figure 1: PagedAttention maps contiguous logical blocks to non-contiguous physical blocks, eliminating external fragmentation.</em>
</p>

## 2. The Impact: Efficiency, Throughput, and Scaling
By decoupling the logical view from physical memory, vLLM transforms both the raw performance of the GPU and the way applications handle complex decoding.

### Core Performance Gains: Memory and Throughput
* **Near-Optimal Memory Usage:** Research indicates that traditional systems typically waste **60% to 80%** of GPU memory due to static over-reservation [^ref-vllm-2023]. vLLM reduces this waste to under **4%**, effectively doubling or tripling the number of concurrent requests a single GPU can handle.
* **Continuous Batching:** Traditional batching waits for the entire batch to finish. vLLM uses **Iteration-level Scheduling**, checking for completed sequences after *every single token*. Completed requests are immediately evicted, and new ones are inserted, ensuring the GPU is never idle.

### Operational Capabilities: Sampling and Caching
Beyond raw speed, PagedAttention enables complex sharing patterns [^ref-vllm-blog] previously too memory-intensive for production use.

* **Parallel Sampling (Intra-Request):** When one request asks for multiple outputs (e.g., `n=5`), vLLM stores the prompt's KV cache exactly once. All generated sequences point back to these same physical blocks, branching only when they begin to generate unique tokens.
<p align="center">
  <img src="/images/inference-2026-vllm/parallel_sampling.gif" width="500">
  <br />
  <em>Figure 2: Parallel sampling in action. Multiple outputs share physical memory for the initial prompt.</em>
</p>

* **Automatic Prefix Caching (Inter-Request):** **Prefix Caching** allows Request B to reuse memory from Request A. In multi-turn conversations or agentic workflows, different requests often share a common system prompt. vLLM caches these blocks across requests, significantly reducing "Time to First Token" (TTFT).
<p align="center">
  <img src="/images/inference-2026-vllm/memory_sharing.gif" width="700">
  <br />
  <em>Figure 3: Shared Prefix Caching across independent requests.</em>
</p>

## 3. Hardware and Model Nuances: The "Block" Reality
The default block size in vLLM is **16 tokens**, a choice driven by hardware constraints.

### GPU Warp Alignment & Throughput
* **The 16-Token Standard:** A **Warp** consists of 32 threads. In vLLM’s kernels, these threads fetch 16 Key and 16 Value vectors in a single coalesced memory transaction, fully saturating GPU bandwidth.
* **The TensorRT-LLM Divergence:** Enterprise engines like TensorRT-LLM often default to **64 or 128-token blocks** [^ref-trtllm]. Larger blocks maximize throughput on H100s by reducing "indirection overhead" (fewer block table lookups) at the cost of higher fragmentation.

### Model Architecture and Memory Variance
The "heaviness" of a block scales with the model's dimensions. For example, at FP16 precision [^ref-llama3-memory]:
* **Llama 3 8B:** A 16-token block consumes **~1.0 MB**.
* **Llama 3 70B:** A 16-token block consumes **~5.2 MB**.

Modern architectures like **DeepSeek-V3** use **Multi-Head Latent Attention (MLA)** to compress KV vectors, meaning 16 tokens can consume significantly less memory even at larger scales.

**Formula for one vLLM block (16 tokens):**
$$\text{Bytes} = 16 \times \text{Layers} \times n_{KV\_heads} \times d_{head} \times \text{Precision\_Bytes} \times 2$$

## 4. Engineering the 'Magic': Production Implementation
The true production strength of vLLM lies in how it manages the gap between the CPU scheduler and the GPU kernels.

### I. CUDA Graph Replay
Launching kernels for every token is expensive. vLLM uses **CUDA Graphs** to "record" and "replay" kernel launches. 
* **Fixed Shapes:** CUDA Graphs require fixed tensor shapes. vLLM captures graphs for specific batch sizes (e.g., 1, 2, 4, 8, 16...). 
* **The Padded Batch:** if a batch contains 7 requests, it is padded to 8 for the forward pass. While this creates a "straggler" effect where one long request keeps a batch slot occupied, Iteration-level scheduling ensures we fill other slots as soon as they open.

### II. The "Swapping" Safety Valve
If VRAM is 100% full and a running request needs a new block, vLLM uses **Swapping**. It preempts the most recent request and moves its KV blocks from GPU VRAM to **CPU RAM**. This prevents Out-of-Memory (OOM) errors at the cost of a significant latency hit.

### III. Synergy with Speculative Decoding
In speculative decoding, a "draft" model predicts tokens that may be rejected by the "target" model. With PagedAttention, rejecting tokens is a simple metadata operation—unmapping physical blocks—rather than a costly memory re-alignment.

---

[^ref-vllm-2023]: Kwon et al., "Efficient Memory Management for Large Language Model Serving with PagedAttention," SOSP 2023. [https://arxiv.org/abs/2309.06180](https://arxiv.org/abs/2309.06180)
[^ref-vllm-blog]: "vLLM: Easy, Fast, and Cheap LLM Serving with PagedAttention," vLLM Blog. [https://vllm.ai/blog/vllm](https://vllm.ai/blog/vllm)
[^ref-trtllm]: NVIDIA, "TensorRT-LLM Documentation: KV Cache Management." [https://nvidia.github.io/TensorRT-LLM/latest/features/kvcache.html](https://nvidia.github.io/TensorRT-LLM/latest/features/kvcache.html)
[^ref-llama3-memory]: "Llama 3 Model Card and Performance Benchmarks," Meta AI Research. [https://github.com/meta-llama/llama3](https://github.com/meta-llama/llama3)
