---
layout: post
title: "vLLM - Revisit"
date: 2026-04-20
categories: [AI, Infrastructure]
tags: [LLM, Inference, Optimization]
description: A technical summary of PagedAttention and memory management in vLLM.
---

*Estimated read time: 10 minutes*

In the world of Large Language Model (LLM) inference, the primary bottleneck isn't just compute—it's memory management. Specifically, the management of the **Key-Value (KV) Cache**. vLLM has emerged as the industry standard by borrowing a classic concept from Operating Systems: **Virtual Memory** [^ref-vllm-2023].

## 1. The Core Innovation: PagedAttention
Traditional inference engines allocate KV cache in large, contiguous blocks. Because we don't know the output length in advance, systems "over-reserve" space, leading to **Internal Fragmentation** (wasted space within a sequence) and **External Fragmentation**.

**PagedAttention** solves this by partitioning the KV cache into small, fixed-size **Physical Blocks**. A **Block Table** maps these logical sequences to physical locations, allowing memory to be allocated on-demand. The following animation from the official vLLM blog illustrates this mapping process.

<p align="center">
  <img src="/images/inference-2026-vllm/animation.gif" width="700">
  <br />
  <em>Figure 1: PagedAttention maps contiguous logical blocks to non-contiguous physical blocks, eliminating external fragmentation.</em>
</p>

## 2. Practical Benefits: Efficiency and Throughput
By decoupling the logical view from physical memory, vLLM provides immediate performance gains for standard serving workloads.

### I. Near-Optimal Memory Usage
Research has shown that traditional systems typically waste **60% to 80%** of GPU memory due to over-reservation [^ref-vllm-2023]. vLLM reduces this waste to under **4%**, allowing for significantly larger batch sizes on the same hardware.

### II. Massively Higher Throughput
By utilizing **Continuous Batching** alongside PagedAttention, vLLM achieves up to **24x higher throughput** than baseline implementations. It eliminates "bubbles" in the pipeline by inserting new requests into a running batch as soon as any single sequence finishes.

## 3. Advanced Memory Sharing: Intra and Inter-Request
PagedAttention enables complex decoding patterns that were previously too memory-intensive by allowing different "views" to point to the same physical data. These diagrams from the vLLM blog highlight the two primary sharing mechanisms.

### I. Parallel Sampling (Intra-Request)
Parallel sampling occurs when one request asks for multiple outputs (e.g., `n=5`). vLLM stores the prompt's KV cache exactly once. All generated sequences point back to these same physical blocks, branching only when they begin to generate unique tokens.

<p align="center">
  <img src="/images/inference-2026-vllm/parallel_sampling.gif" width="500">
  <br />
  <em>Figure 2: Parallel sampling in action. Multiple outputs share physical memory for the initial prompt.</em>
</p>

### II. Automatic Prefix Caching (Inter-Request)
**Prefix Caching** allows Request B to reuse memory from Request A. In multi-turn conversations or agentic workflows, different requests often share a common system prompt. vLLM caches these blocks across requests, significantly reducing "Time to First Token" (TTFT) and total VRAM usage.

<p align="center">
  <img src="/images/inference-2026-vllm/memory_sharing.gif" width="700">
  <br />
  <em>Figure 3: Shared Prefix Caching across independent requests.</em>
</p>

## 4. Why Block Size Matters: The 16-Token Standard
The default block size in vLLM is **16 tokens**, a choice driven by hardware constraints.

* **GPU Warp Alignment:** A **Warp** consists of 32 threads. In vLLM’s kernels, these threads fetch 16 Key and 16 Value vectors in a single coalesced memory transaction, fully saturating GPU bandwidth.
* **The TensorRT-LLM Divergence:** Enterprise engines like TensorRT-LLM often default to **64 or 128-token blocks** [^ref-trtllm]. While larger blocks maximize throughput on H100s by reducing "indirection overhead" (fewer block table lookups), they increase the fragmentation waste that small blocks help avoid.
* **Memory Variance:** 16 tokens do not represent a fixed byte size. A block for Llama 3 70B (~5.2 MB) is much "heavier" than one for Llama 3 8B (~1 MB). Sticking to 16 tokens allows for surgical **Copy-on-Write** (CoW) branching regardless of the model's footprint.

## 5. Synergy with Advanced Architectures
Block-based management is particularly powerful for modern inference strategies:

### Speculative Decoding
In speculative decoding, a "draft" model predicts tokens that may be rejected by the "target" model. Rejecting tokens becomes a simple metadata operation—unmapping physical blocks—rather than a costly memory re-alignment.

### Multi-Token Prediction (MTP)
Architectures that predict multiple future paths (like Medusa or DeepSeek-V3) create a "tree" of tokens. vLLM’s block-based logic handles this naturally, following the most likely branch while discarding others without fragmentation.

## 6. Memory Calculation Nuances
In modern models, the KV cache size is often decoupled from the total hidden dimension ($d_{model}$):

* **Grouped-Query Attention (GQA):** Only a fraction of KV heads exist compared to Query heads (e.g., Llama 3 70B has an 8:1 ratio).
* **Multi-Head Latent Attention (MLA):** Compressed KV vectors (like in DeepSeek-V3) mean 16 tokens consume significantly less memory.

**Formula for one vLLM block (16 tokens):**
$$\text{Bytes} = 16 \times \text{Layers} \times n_{KV\_heads} \times d_{head} \times \text{Precision\_Bytes} \times 2$$

---

[^ref-vllm-2023]: Kwon et al., "Efficient Memory Management for Large Language Model Serving with PagedAttention," SOSP 2023. [https://arxiv.org/abs/2309.06180](https://arxiv.org/abs/2309.06180)
[^ref-vllm-blog]: "vLLM: Easy, Fast, and Cheap LLM Serving with PagedAttention," vLLM Blog. [https://vllm.ai/blog/vllm](https://vllm.ai/blog/vllm)
[^ref-trtllm]: NVIDIA, "TensorRT-LLM Documentation: KV Cache Management." [https://nvidia.github.io/TensorRT-LLM/latest/features/kvcache.html](https://nvidia.github.io/TensorRT-LLM/latest/features/kvcache.html)
