---
layout: post
title: "vLLM - Revisit"
date: 2026-04-20
categories: [AI, Infrastructure]
tags: [LLM, Inference, Optimization]
description: A technical summary of PagedAttention and memory management in vLLM.
---

*Estimated read time: 8 minutes*

In the world of Large Language Model (LLM) inference, the primary bottleneck isn't just compute—it's memory management. Specifically, the management of the **Key-Value (KV) Cache**. vLLM has emerged as the industry standard by borrowing a classic concept from Operating Systems: **Virtual Memory** [^ref-vllm-2023].

## 1. The Core Innovation: PagedAttention
Traditional inference engines allocate KV cache in large, contiguous blocks. Because we don't know the output length in advance, systems "over-reserve" space, leading to **Internal Fragmentation** (wasted space within a sequence) and **External Fragmentation**.

**PagedAttention** solves this by partitioning the KV cache into small, fixed-size **Physical Blocks**. A **Block Table** maps these logical sequences to physical locations, allowing memory to be allocated on-demand.

<p align="center">
  <img src="https://vllm.ai/static/figures/paged_attention.png" width="800">
  <br />
  <em>Figure 1: PagedAttention maps contiguous logical blocks to non-contiguous physical blocks, eliminating external fragmentation.</em>
</p>

## 2. Practical Benefits of vLLM
By decoupling the logical view from physical memory, vLLM provides three transformative benefits for production environments:

### I. Near-Optimal Memory Usage
Traditional systems often waste **60% to 80%** of GPU memory. vLLM reduces this waste to under **4%**, allowing for significantly larger batch sizes on the same hardware.

### II. Massively Higher Throughput
By utilizing **Continuous Batching** alongside PagedAttention, vLLM achieves up to **24x higher throughput** than baseline implementations. This is achieved by inserting new requests into a running batch at every generation step.

### III. Memory Sharing & Prefix Caching
Because memory is managed in blocks, multiple requests can share the same physical memory for a shared prefix (like a system prompt). This is crucial for **Parallel Sampling** and **Beam Search**, where multiple sequences "fork" from the same history.

<p align="center">
  <img src="https://vllm.ai/static/figures/sharing.png" width="800">
  <br />
  <em>Figure 2: Memory sharing in action. Multiple sequences share physical blocks for the prefix, branching only when they diverge.</em>
</p>

## 3. Why Block Size Matters: The 16-Token Standard
The default block size in vLLM is **16 tokens**. This choice is driven by a trade-off between memory waste and hardware efficiency.

* **GPU Warp Alignment:** A **Warp** consists of 32 threads. In vLLM’s kernels, these 32 threads can fetch 16 Key and 16 Value vectors in a single coalesced memory transaction, fully saturating GPU bandwidth.
* **The TensorRT-LLM Divergence (128 Tokens):** Enterprise engines like TensorRT-LLM often use **128-token blocks**. While the Warp size is the same, larger blocks maximize throughput on H100s by reducing "indirection overhead"—the number of times the system must consult the Block Table.
* **Memory Variance:** 16 tokens do not represent a fixed byte size. A block for Llama 3 70B (~5.2 MB) is much "heavier" than one for Llama 3 8B (~1 MB). Sticking to 16 tokens allows for surgical **Copy-on-Write** (CoW) branching, even as model sizes scale.

## 4. Synergy with Advanced Architectures
Block-based management is particularly powerful for modern inference strategies:

### Speculative Decoding
In speculative decoding, a "draft" model predicts tokens that may be rejected by the "target" model. With PagedAttention, rejecting tokens is a simple metadata operation—unmapping physical blocks—rather than a costly memory re-alignment.

### Multi-Token Prediction (MTP)
Architectures that predict multiple future paths (like Medusa or DeepSeek-V3) create a "tree" of tokens. vLLM’s block-based logic handles this naturally, allowing the system to follow the most likely branch while discarding others without memory fragmentation.

## 5. Memory Calculation Nuances
In modern models, the KV cache size is often decoupled from the total hidden dimension ($d_{model}$):

* **Grouped-Query Attention (GQA):** Only a fraction of KV heads exist compared to Query heads (e.g., Llama 3 70B has an 8:1 ratio).
* **Multi-Head Latent Attention (MLA):** Compressed KV vectors (like in DeepSeek-V3) mean 16 tokens consume significantly less memory.

**Formula for one vLLM block (16 tokens):**
$$\text{Bytes} = 16 \times \text{Layers} \times n_{KV\_heads} \times d_{head} \times \text{Precision\_Bytes} \times 2$$

---

[^ref-vllm-2023]: Kwon et al., "Efficient Memory Management for Large Language Model Serving with PagedAttention," SOSP 2023. [https://arxiv.org/abs/2309.06180](https://arxiv.org/abs/2309.06180)
