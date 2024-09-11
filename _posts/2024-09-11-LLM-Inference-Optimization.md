# Inference Optimization of Large Transformer Model

TLDR:

[TOC]

## Large Transformer Model Inference Overview
* Prefilling (context processing) + Generation

### Challenges of Inferencing Large Transformer Model
* Context phase: heavy computation, handling super-long context
* Generation phase: KV cache

## Algorithmic Optimization
Optimizations that might impact model quality

### General Methodologies

#### Quantization
* Weights-only Quantization vs Activation Quantization
  * Weights Quantization - only optimize for storage, weights are converted back to FP16 at computation time
  * Activation Quantization - quantize to int8, int4, and recently Nvidia hardware had support for [FP8](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/examples/fp8_primer.html)
* Post-training quantization (PTQ) vs Quantization-aware training (QAT)
  * PTQ
  * QAT

#### Distillation
* Knowledge transfer to a smaller student model
* The key is to have the student model learn from both ground truth labels and the softened softmax output of the teacher model (softmax with a higher temperature)

#### Pruning & Sparsity
* TBD

### Model Architecture Improvements with Sparsity Considerations

#### MQA and GQA
To reduce the KV cache size, the idea is to share the same key and value among all or a group of heads.
* In the scenario of all heads sharing the same key and values, but only using different queries, it's called multi-query attention (MQA).
* research has discovered that MQA is too aggressive and the model performs degrades. A sweet spot is having a group of queries (instead of all) sharing the same key and value, which is a tradeoff between KV cache size optimization and model quality.
  * [Llama-3](https://ai.meta.com/blog/meta-llama-3/) uses GQA.

#### Mixture of Expert (MOE)


## Implementation / System Optimization
### PageAttention
Use a page table to make use of fragmented memory.

### StreamingLLM
To support super-long context, an artificial **attention sink** was used to preserve model quality.

### FlashAttention
It was discovered that the majority of time consumed during the context phase is I/O. FlashAttention uses the idea of tiling and only loads part of the caches when computing attention scores to ensure more computations are conducted in high-speed SRAM and achieve a 4x speedup without impacting model accuracy.

Flash attention is an *exact optimization*, meaning the computation will be the same as the conventional attention. It only optimizes data access patterns (through tiling) to reduce the I/O overhead.

### Speculative Decoding


### References
