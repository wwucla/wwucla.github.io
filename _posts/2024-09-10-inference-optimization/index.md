# Large Transformer Model - Inference Optimization

TLDR:

<!-- TOC start (generated with https://github.com/derlin/bitdowntoc) -->

- [Large Transformer Model Inference Overview](#large-transformer-model-inference-overview)
  * [Challenges of Inferencing Large Transformer Model](#challenges-of-inferencing-large-transformer-model)
- [Algorithmic Optimization](#algorithmic-optimization)
  * [General Methodologies](#general-methodologies)
    + [Quantization](#quantization)
    + [Distillation](#distillation)
    + [Pruning & Sparsity](#pruning--sparsity)
  * [Model Architecture Improvements with Sparsity Considerations](#model-architecture-improvements-with-sparsity-considerations)
    + [MQA and GQA](#mqa-and-gqa)
    + [Mixture of Expert (MOE)](#mixture-of-expert-moe)
- [Implementation / System Optimization](#implementation--system-optimization)
  * [PageAttention](#pageattention)
  * [StreamingLLM](#streamingllm)
  * [FlashAttention](#flashattention)
  * [Speculative Decoding](#speculative-decoding)
  * [References](#references)

<!-- TOC end -->


## Large Transformer Model Inference Overview
The majority of the LLMs today use different variants of transformer architecture. Such models take the context (or more precisely, tokens) as input, and keep generating the next tokens until the output is a special <`end`> token, which terminates the generation. 

### Two-phase process
LLM inference typically runs in two phases:
* Prefill Phase (aka initialization phase)
LLM processes input tokens in this phase and computes keys and values at each decoder layer (KV cache). Given all input tokens are available at once, it involves large-scale matrix-matrix operations that can be highly parallelized, especially when the input context is long.
* Decode Phase (aka generation phase)
LLM uses the KV cache to compute the next output token, which is later used as input to generate the next output. During each token decode/generation step, the keys and values are stored in the KV cache therefore they don't have to be recomputed. Despite the autoregressive decode process is sequential, it still involves a large amount of matrix-vector operations, which can be parallelized. 

Prefill and decode phases are typically implemented separately due to distinct computation patterns and they can be optimized differently. A more complicated LLM inference server (Figure 1: ) involves a query queue scheduler, an inference engine that handles dynamic batching and the actual inference work. It utilizes GPU or other custom accelerators to speed up the computation.

<p align="center">
    <img src="images/llm_inference_server_arch.png" width="800" class="center">
 
    <em>Figure 1: Typical Architecture of LLM Inference Servers and Engines</em>
</p>
<!-- ![server_arch](images/llm_inference_server_arch.png) -->

### Challenges of Inferencing Large Transformer Model
There are multiple challenges around LLM inference:
* 

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
