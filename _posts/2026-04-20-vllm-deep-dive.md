---
layout: post
title: "vLLM - Revisit"
date: 2026-04-20
categories: [AI, Infrastructure]
tags: [LLM, Inference, Optimization]
description: A technical summary of PagedAttention and memory management in vLLM.
---

*Estimated read time: 6 minutes*

In the world of Large Language Model (LLM) inference, the primary bottleneck isn't just compute—it's memory management. Specifically, the management of the **Key-Value (KV) Cache**. vLLM has emerged as the industry standard by borrowing a classic concept from Operating Systems: **Virtual Memory** [^ref-vllm-2023].

## 1. The Core Innovation: PagedAttention
Traditional inference engines allocate KV cache in large, contiguous blocks. Because we don't know the output length in advance, systems "over-reserve" space, leading to **Internal Fragmentation** (wasted space within a sequence) and **External Fragmentation**.

**PagedAttention** solves this by:
- Partitioning the KV cache into small, fixed-size **Physical Blocks**.
- Using a **Block Table** to map **Logical Blocks** (the model's view) to these physical locations.
- Allocating memory on-demand, achieving near-zero waste and allowing for significantly larger batch sizes.

## 2. Why Block Size Matters
The "Golden Ratio" for block size in the standard vLLM implementation is **16 tokens**.

* **The Trade-off:** Smaller blocks minimize fragmentation but increase the overhead of the Block Table. Larger blocks (e.g., 64 or 128) are more hardware-friendly for coalesced memory reads but lead to higher waste.
* **Framework Divergence:** vLLM defaults to 16 for flexibility across diverse hardware, while frameworks like TensorRT-LLM often use 64 or 128 to maximize throughput on data-center GPUs (H100/A100).

## 3. Synergy with Advanced Architectures
Block-based management is particularly powerful for modern inference strategies:

### Speculative Decoding
In speculative decoding, a "draft" model predicts tokens that may be rejected by the "target" model. With PagedAttention, rejecting tokens is a simple metadata operation—unmapping physical blocks—rather than a costly memory re-alignment.

### Multi-Token Prediction (MTP) & Tree Search
Architectures that predict multiple future paths (like Medusa or DeepSeek-V3) create a "tree" of tokens. vLLM uses **Copy-on-Write** logic, allowing multiple branches to share the same physical blocks for the shared prefix, only branching off into new blocks when paths diverge.

## 4. Memory Calculation Nuances
It is a common misconception to use the **Model Dimension** ($d_{model}$) to calculate block memory. In modern models, the KV cache size is often decoupled from the total hidden dimension:

* **Grouped-Query Attention (GQA):** Only a fraction of KV heads exist compared to Query heads (e.g., Llama 3 70B has an 8:1 ratio).
* **Multi-Head Latent Attention (MLA):** Compressed KV vectors mean 16 tokens might consume significantly less memory than in a standard transformer.

**Formula for one vLLM block (16 tokens):**
$$\text{Bytes} = 16 \times \text{Layers} \times n_{KV\_heads} \times d_{head} \times \text{Precision\_Bytes} \times 2$$

---

[^ref-vllm-2023]: Kwon et al., "Efficient Memory Management for Large Language Model Serving with PagedAttention," SOSP 2023. [https://arxiv.org/abs/2309.06180](https://arxiv.org/abs/2309.06180)
