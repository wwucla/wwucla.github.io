# Large Transformer Model - Inference Optimization

**TL;DR**: This note provides a comprehensive overview of LLM inference, including challenges it presents and potential solutions through algorithmic optimization and system implementation improvements.

<!-- TOC start (generated with https://github.com/derlin/bitdowntoc) -->

- [Overview](#overview)
   * [Two-phase process](#two-phase-process)
   * [Challenges of Inferencing Large Transformer Model](#challenges-of-inferencing-large-transformer-model)
- [Algorithmic Optimization](#algorithmic-optimization)
   * [Quantization](#quantization)
      + [Weights-only Quantization vs Activation Quantization](#weights-only-quantization-vs-activation-quantization)
      + [Post-training quantization (PTQ) vs Quantization-aware training (QAT)](#post-training-quantization-ptq-vs-quantization-aware-training-qat)
      + [SOTA Developments - SmoothQuant](#sota-developments-smoothquant)
      + [SOTA Developments - AWQ](#sota-developments-awq)
   * [Knowledge Distillation](#knowledge-distillation)
   * [Pruning & Sparsity](#pruning-sparsity)
   * [Transformer Model Architecture Optimization](#transformer-model-architecture-optimization)
      + [Multi-Query Attention (MQA) and Grouped-Query Attention (GQA)](#multi-query-attention-mqa-and-grouped-query-attention-gqa)
      + [Mixture of Expert (MOE)](#mixture-of-expert-moe)
- [Implementation / System Optimization](#implementation-system-optimization)
   * [vLLM (PagedAttention)](#vllm-pagedattention)
   * [StreamingLLM](#streamingllm)
   * [FlashAttention](#flashattention)
   * [Speculative Decoding](#speculative-decoding)
- [References](#references)

<!-- TOC end -->


<!-- TOC --><a name="overview"></a>
## Overview
Most contemporary LLMs are based on the transformer architecture. These models process input text sequentially, token by token. The model generates subsequent tokens until a designated termination token, such as `<end>`, is produced, signaling the completion of the output sequence.

<!-- TOC --><a name="two-phase-process"></a>
### Two-phase process
LLM inference is generally divided into two primary phases:
* **Prefill Phase** (aka initialization phase): This phase involves processing the entire input sequence and constructing key-value (KV) caches for each decoder layer. Given the availability of all input tokens, this phase is amenable to efficient parallelization, particularly for long input contexts.
* **Decode Phase** (aka generation phase): LLM iteratively generates output tokens, using the previously generated tokens and the KV caches to compute the next token. While the decoding process is sequential, it still involves matrix-vector operations that can be parallelized.

<p align="center">
  <img src="/images/inference-optimization/two_phases.webp" width="600"><br />
  Figure 1: Prefill and Decode phases for LLM inference [source: https://www.adyen.com/knowledge-hub/llm-inference-at-scale-with-tgi]
</p>

A typical LLM inference server architecture[^ref-llm-arch] is illustrated in Figure 1. It includes:
* **Query Queue Scheduler**: manages incoming queries and optimizes batching for efficient inference.
* **Inference Engine**: handles dynamic batching and orchestrates the prefill and decode phases. It employs GPUs or other specialized hardware to accelerate computationally intensive operations.

Due to the distinct computational patterns of the prefill and decode phases, they are often optimized separately. This allows for tailored hardware and software optimizations to maximize performance.

<p align="center">
  <img src="/images/inference-optimization/llm_server_arch.png" width="900"><br />
  Figure 2: Typical Architecture of LLM Inference Servers and Engines
</p>

<!-- TOC --><a name="challenges-of-inferencing-large-transformer-model"></a>
### Challenges of Inferencing Large Transformer Model
There are multiple challenges around LLM inference:
* **Heave computation** in both prefill and decode phase
* **Storage challenge of KV cache**: storage requirement `batch_size * n_layers * (d_model/n_heads) * n_kv_heads * 2 (K and V) * 2 (sizeof FP16) * seq_len`, propotional to `seq_len`.
* **Handling super-long context**: The capability to handle extensive sequences is essential for applications such as summarizing lengthy documents and RAG. However, this requirement often strains both storage capacity and computational resources.
* **Efficient KV cache management when serving multiple queries**

The following sections below will discuss different optimizations for LLM inference.


<!-- TOC --><a name="algorithmic-optimization"></a>
## Algorithmic Optimization
This section explores optimizations that modify the LLM algorithm to enhance inference efficiency. We'll begin with general approaches applicable to many ML architectures and how they are applied to transformer models, including:

* Quantization: Reducing model size and precision by using lower-bit weights and/or activations.
* Distillation: Training a smaller "student" model to mimic the behavior of a larger "teacher" model.
* Pruning: Introducing sparsity by removing unnecessary connections or parameters.

Then, we'll delve into optimizations tailored to transformer models, discussing several variants designed for more efficient inference.

<!-- TOC --><a name="quantization"></a>
### Quantization
<!-- TOC --><a name="weights-only-quantization-vs-activation-quantization"></a>
#### Weights-only Quantization vs Activation Quantization
* **Weights-only Quantization (WOQ)** 
  * WOQ focuses on quantizing the model weights. It reduces the model size, leading to faster loading time and lower memory usage during inference. Typically used precision formats are INT8 or INT4 for weights, while activations remain in FP16 for better accuracy. Recently Nvidia hardware added support for [FP8](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/examples/fp8_primer.html), providing another alternative for quantization.
  * WOQ can be a good trade-off between model size and accuracy, especially for large language models (LLMs) where memory bandwidth is a major bottleneck.
* **Activation Quantization (AQ)**
  * AQ quantizes both the weights and activations of the model. It can potentially achieve higher compression ratios compared to WOQ. However, it's more prone to accuracy degradation due to the presence of **outliers in activations**[^ref-smoothquant], which can be amplified during quantization.
  * Requires careful selection of quantization techniques to minimize accuracy loss. (to be covered later in this section)

 In summary, **WOQ** is generally preferred for LLMs due to its better balance of accuracy and efficiency. **AQ** can be beneficial for certain tasks if implemented carefully, but requires more fine-tuning to avoid accuracy drops.

<!-- TOC --><a name="post-training-quantization-ptq-vs-quantization-aware-training-qat"></a>
#### Post-training quantization (PTQ) vs Quantization-aware training (QAT)
* **Post-training quantization (PTQ)** is a straightforward and cost-effective method that directly converts the weights of a pre-trained model to lower precision without requiring any additional training. It reduces the model's size and improves inference speed.
* **Quantization-aware training (QAT)** introduced by [Jacob et al. 2017](https://arxiv.org/abs/1712.05877)[^ref-qat], allows for training models with lower-precision weights and activations during the forward pass. This reduces memory usage and improves inference speed. However, the backward pass, which calculates gradients for weight updates, still uses full precision to maintain accuracy. While QAT typically leads to higher quality quantized models compared to post-training quantization (PTQ), it requires a more complex setup. Fortunately, mainstream ML platforms like TensorFlow offer support for both QAT and PTQ (e.g. [QAT support in Tensorflow](https://www.tensorflow.org/model_optimization/guide/quantization/training)).
 

<!-- TOC --><a name="sota-developments-smoothquant"></a>
#### SOTA Developments - SmoothQuant
**SmoothQuant[^ref-smoothquant]** ([Xiao et al. 2023](https://arxiv.org/abs/2211.10438)) discovered that outliers in activations become more prevalent as the model size grows. These outliers can significantly degrade quantization performance (illustrated in the figure below), leading to higher quantization errors and potentially impacting the quality of the quantized model.  In contrast, the weights have fewer outliers and are generally easier to quantize.

<p align="center">
  <img src="/images/inference-optimization/smoothquant-error-outlier.png" width="400"><br />
  Figure 2: Model size vs Accuracy of quantized model [from SmoothQuant paper]
</p>

The key idea of SmoothQuant is to migrate part of the quantization challenges from activation to weights, which smooths out the systematic outliers in activation, making both weights and activations easy to quantize.

<p align="center">
  <img src="/images/inference-optimization/smoothquant_intuition.png" width="800"><br />
  Figure 3: SmoothQuant Intuition [from SmoothQuant paper]
</p>

<!-- TOC --><a name="sota-developments-awq"></a>
#### SOTA Developments - AWQ
**AWQ** ([Lin et al, 2024](https://arxiv.org/abs/2306.00978))[^ref-awq]


<!-- TOC --><a name="knowledge-distillation"></a>
### Knowledge Distillation
The high-level idea of knowledge distillation ([Hinton et al, 2015](https://arxiv.org/abs/1503.02531)) is to transfer knowledge from a cumbersome teacher model to a smaller student mode. Illustrated in the figure below.

<p align="center">
  <img src="/images/inference-optimization/kd_framework.png" width="600"><br />
  Figure 3: Knowledge Distillation Framework [source: https://arxiv.org/abs/2006.05525]
</p>

The key idea of knowledge distillation is well articulated in the following paragraph quoted from the orignal paper:

* *"We found that a better way is to simply use a **weighted average of two different objective functions**. The first objective function is the cross entropy with the soft targets and this cross entropy is computed using the same high temperature in the softmax of the distilled model as was used for generating the soft targets from the cumbersome model. The second objective function is the cross entropy with the correct labels. This is computed using exactly the same logits in softmax of the distilled model but at a temperature of 1. We found that the best results were generally obtained by using a **condiderably lower weight on the second objective function**"*

It uses a higher temperature to soften the learning objective (the relationship between temperature and actual labels is illustrated in the figure below).
<p align="center">
  <img src="/images/inference-optimization/softmax.gif" width="300"><br />
  Figure 4: Visualizing the Effects of Temperature Scaling [source: https://medium.com/@harshit158/softmax-temperature-5492e4007f71]
</p>

Denoting the logits before the final softmax layer as **$z_t$** and **$z_s$** for teacher and student models, label as **$y$**, temperature as **$T$**, the learning objective described in the original paper can be represented as:

$$L_\text{Distillation} = L_\text{CE}(\text{Softmax}(z_t, T), \text{Softmax}(z_s, T)) + \lambda L_\text{CE}(\text{Softmax}(z_s, 1), y)$$



<!-- TOC --><a name="pruning-sparsity"></a>
### Pruning & Sparsity
TODO - add more details

<!-- TOC --><a name="transformer-model-architecture-optimization"></a>
### Transformer Model Architecture Optimization

<!-- TOC --><a name="multi-query-attention-mqa-and-grouped-query-attention-gqa"></a>
#### Multi-Query Attention (MQA) and Grouped-Query Attention (GQA)
To reduce the KV cache size, the idea is to share the same key and value among all or a group of heads.
* In the scenario of all heads sharing the same key and values, but only using different queries, it's called multi-query attention (MQA) [^ref-mqa].
* research has discovered that MQA is too aggressive and the model performs degrades. Grouped-Query Attention (GQA) [^ref-gqa] was proposed to have a group of queries (instead of all) sharing the same key and value, which is a tradeoff between KV cache size optimization and model quality. 
  * Llama-3[^ref-llama3] ([Dubey et al 2024](https://arxiv.org/abs/2407.21783)) uses GQA.

<!-- TOC --><a name="mixture-of-expert-moe"></a>
#### Mixture of Expert (MOE)


<!-- TOC --><a name="implementation-system-optimization"></a>
## Implementation / System Optimization
<!-- TOC --><a name="vllm-pagedattention"></a>
### vLLM (PagedAttention)
[Kwon et al 2023](https://arxiv.org/abs/2309.06180) [^ref-vllm]

Use a page table to make use of fragmented memory.


<!-- TOC --><a name="streamingllm"></a>
### StreamingLLM
[Xiao et al 2023](https://arxiv.org/abs/2309.17453) [^ref-streamingllm]

To support super-long context, an artificial **attention sink** was used to preserve model quality.

<!-- TOC --><a name="flashattention"></a>
### FlashAttention
* FlashAttention ([Dao et al 2022](https://arxiv.org/abs/2205.14135)) [^ref-flashattention]
* FlashAttention2 ([Dao et al 2022](https://arxiv.org/abs/2307.08691)) [^ref-flashattention2]
* FlashAttention3 ([Shah et al 2024](https://arxiv.org/abs/2407.08608)) [^ref-flashattention3]

It was discovered that the majority of time consumed during the context phase is I/O. FlashAttention uses the idea of tiling and only loads part of the caches when computing attention scores to ensure more computations are conducted in high-speed SRAM and achieve a 4x speedup without impacting model accuracy.

Flash attention is an *exact optimization*, meaning the computation will be the same as the conventional attention. It only optimizes data access patterns (through tiling) to reduce the I/O overhead.

<!-- TOC --><a name="speculative-decoding"></a>
### Speculative Decoding

[Leviathan et al 2022](https://arxiv.org/abs/2211.17192) [^ref-spec-decoding]

Similar to the idea of speculative execution in a pipeline, here it uses a smaller LLM model to predict the next few tokens, and apply the larger model to validate the quality of the predictions. Because larger models process a group of tokens instead of one by one, there is more potential to optimize for runtime. On T5-XXL, it achieves a 2X-3X acceleration compared to the standard T5X implementation, with identical outputs.

<!-- TOC --><a name="references"></a>
## References

[^ref-llm-arch]: Ekin Karabulut, Omer Dayan. "[What it means to serve an LLM and which serving technology to choose from](https://www.run.ai/blog/serving-large-language-models)", 2024
[^ref-smoothquant]: Xiao, Guangxuan, et al. "[SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models](https://arxiv.org/abs/2211.10438)." International Conference on Machine Learning. PMLR, 2023.
[^ref-awq]: Lin, Ji, et al. "[AWQ: Activation-aware Weight Quantization for On-Device LLM Compression and Acceleration](https://arxiv.org/abs/2306.00978)." Proceedings of Machine Learning and Systems 6 (2024): 87-100.
[^ref-qat]: Jacob, Benoit, et al. "[Quantization and training of neural networks for efficient integer-arithmetic-only inference](https://arxiv.org/abs/1712.05877)." Proceedings of the IEEE conference on computer vision and pattern recognition. 2018.
[^ref-vllm]: Kwon, Woosuk, et al. "[Efficient memory management for large language model serving with paged attention](https://arxiv.org/abs/2309.06180)." Proceedings of the 29th Symposium on Operating Systems Principles. 2023.
[^ref-streamingllm]: Xiao, Guangxuan, et al. "[Efficient streaming language models with attention sinks](https://arxiv.org/abs/2309.17453)." arXiv preprint arXiv:2309.17453 (2023).
[^ref-flashattention]: Dao, Tri, et al. "[Flashattention: Fast and memory-efficient exact attention with io-awareness](https://arxiv.org/abs/2205.14135)." Advances in Neural Information Processing Systems 35 (2022): 16344-16359.
[^ref-flashattention2]: Dao, Tri. "[Flashattention-2: Faster attention with better parallelism and work partitioning](https://arxiv.org/abs/2307.08691)." arXiv preprint arXiv:2307.08691 (2023).
[^ref-flashattention3]: Shah, Jay, et al. "[Flashattention-3: Fast and accurate attention with asynchrony and low-precision](https://arxiv.org/abs/2407.08608)." arXiv preprint arXiv:2407.08608 (2024).
[^ref-spec-decoding]: Leviathan, Yaniv, Matan Kalman, and Yossi Matias. "[Fast inference from transformers via speculative decoding](https://arxiv.org/abs/2211.17192)." International Conference on Machine Learning. PMLR, 2023.
[^ref-llama3]: Dubey, Abhimanyu, et al. "[The llama 3 herd of models](https://arxiv.org/abs/2407.21783)." arXiv preprint arXiv:2407.21783 (2024).
[^ref-mqa]: Shazeer, Noam. "[Fast transformer decoding: One write-head is all you need](https://arxiv.org/abs/1911.02150)." arXiv preprint arXiv:1911.02150 (2019).
[^ref-gqa]: Ainslie, Joshua, et al. "[Gqa: Training generalized multi-query transformer models from multi-head checkpoints](https://arxiv.org/abs/2305.13245)." arXiv preprint arXiv:2305.13245 (2023).
