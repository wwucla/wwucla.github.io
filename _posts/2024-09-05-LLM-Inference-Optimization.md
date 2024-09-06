## Inference Optimization of Large Transformer Model

TLDR:

[TOC]

### Challenges of Inferencing Large Transformer Model
* Prefilling (context processing) + Generation
* Challenges of prefilling: heavy computation. When the context sequence is long, data fetch could be even more time-consuming than computation.
    * Typical optimization: Flash attention. Flash attention is an *exact optimization*, meaning the computation will be the same as the conventional attention. It only optimizes data access patterns (through tiling) to reduce the I/O overhead.
* Challenges of Generation: KV cache
    * typical optimization, using multi-query attention (MQA) or group-query attention (GQA) instead of multi-head attention. while MQA could impact the model quality, GQA is a sweet spot that gets decent model quality with reduced KV cache size.
    * [Llama-3](https://ai.meta.com/blog/meta-llama-3/) uses GQA.

### Typical Implementation of Large Transformer Model

Optimizations that might impact model quality

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

#### Mixture of Expert (MOE)


### References
