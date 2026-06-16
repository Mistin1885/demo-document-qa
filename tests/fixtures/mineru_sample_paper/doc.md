<Page 1>
# Adaptive Retrieval-Augmented Generation for Scientific Document QA

Alice Researcher<sup>1</sup>, Bob Scholar<sup>2</sup>, Carol Scientist<sup>1,3</sup>

<sup>1</sup>University of AI, <sup>2</sup>Institute of NLP, <sup>3</sup>Lab for Information Retrieval

## Abstract

We present AdaptRAG, a novel framework for retrieval-augmented generation that dynamically adjusts retrieval granularity based on query complexity. Our approach demonstrates significant improvements over baseline systems on multiple benchmarks, achieving state-of-the-art performance while reducing computational overhead.

## 1 Introduction

Large language models have demonstrated remarkable capabilities in natural language understanding and generation tasks. However, these models suffer from hallucination and outdated knowledge problems that limit their applicability in knowledge-intensive domains. Retrieval-augmented generation (RAG) addresses this limitation by conditioning model outputs on retrieved evidence from external knowledge bases.
</Page 1>

<Page 2>
## 2 Related Work

Prior work on retrieval-augmented generation spans multiple research directions. DPR (Karpukhin et al., 2020) introduced dense passage retrieval using dual-encoder architectures. RAG (Lewis et al., 2020) combined retrieval with sequence-to-sequence generation. More recent work has focused on improving retrieval quality and generation faithfulness.

## 3 Methodology

### 3.1 Adaptive Retrieval Module

The adaptive retrieval module dynamically selects retrieval granularity based on a query complexity score. Given a query q, we compute a complexity score c(q) using a lightweight classifier. The granularity level g is then determined by thresholding c(q) against learned boundaries.

### 3.2 Generation with Retrieved Context

Given the retrieved context C = {c_1, ..., c_k}, the generation model produces an answer conditioned on both the query and the retrieved passages. We employ a cross-attention mechanism to weight the retrieved context appropriately.
</Page 2>

<Page 3>
![Figure 1](images/adaptrag_p3_a1b2c3d4.jpg)

Figure 1: Overview of the AdaptRAG framework. The query complexity classifier routes queries to different retrieval granularities.

Figure 1 illustrates the overall architecture of our proposed system. The input query first passes through the complexity classifier, which outputs a granularity level. Based on this level, the retrieval module queries the knowledge base at different levels of granularity.

$$
P(y|q, C) = \prod_{t=1}^{T} P(y_t | y_{<t}, q, C)
$$
</Page 3>

<Page 4>
Table 1: Performance comparison on NaturalQuestions and TriviaQA benchmarks.

| Model | NQ EM | TQA EM | Latency (ms) |
|-------|-------|--------|--------------|
| DPR + FiD | 51.4 | 67.6 | 230 |
| RAG-Token | 44.5 | 56.8 | 180 |
| AdaptRAG (ours) | **56.2** | **72.1** | 145 |

Table 1 shows the performance comparison. AdaptRAG achieves state-of-the-art results on both benchmarks while significantly reducing inference latency compared to DPR+FiD.
</Page 4>

<Page 5>
## 4 Experiments

We evaluate AdaptRAG on three standard QA benchmarks: NaturalQuestions (NQ), TriviaQA (TQA), and WebQuestions (WQ). Our experiments demonstrate the effectiveness of the adaptive retrieval strategy across different query types and difficulty levels.

### 4.1 Experimental Setup

We use Wikipedia as our knowledge base with 21 million 100-word passages. Dense retrieval uses a pre-trained DPR encoder fine-tuned on NQ training data. The generation model is T5-large (770M parameters) fine-tuned on each downstream task.

### 4.2 Main Results

The main results are reported in Table 1. AdaptRAG outperforms all baseline methods on both NQ and TQA datasets. The adaptive strategy provides the most benefit on complex multi-hop questions, where fine-grained retrieval is most important.
</Page 5>

<Page 6>
## References

[1] Karpukhin, V., Oguz, B., Min, S., Lewis, P., Wu, L., Edunov, S., Chen, D., and Yih, W. (2020). Dense passage retrieval for open-domain question answering. EMNLP.

[2] Lewis, P., Perez, E., Piktus, A., Petroni, F., Karpukhin, V., Goyal, N., Kuttler, H., Lewis, M., Yih, W., Rocktaschel, T., Riedel, S., and Kiela, D. (2020). Retrieval-augmented generation for knowledge-intensive NLP tasks. NeurIPS.

[3] Izacard, G. and Grave, E. (2021). Leveraging passage retrieval with generative models for open domain question answering. EACL.

[4] Kwiatkowski, T., Palomaki, J., Redfield, O., Collins, M., Parikh, A., Alberti, C., Epstein, D., Polosukhin, I., Devlin, J., Lee, K., et al. (2019). Natural questions: A benchmark for question answering research. TACL.

[5] Joshi, M., Choi, E., Weld, D., and Zettlemoyer, L. (2017). TriviaQA: A reading comprehension dataset containing Wikipedia and web search queries. ACL.
</Page 6>

<Page 7>
## Appendix A Additional Experiments

This appendix provides additional experimental results and ablation studies that complement the main paper. We explore sensitivity of the model to hyperparameter choices and provide full result tables for all evaluation benchmarks.
</Page 7>

<Page 8>
### A.1 Ablation Study

We conduct ablation experiments to understand the contribution of each component. Removing the complexity classifier reduces NQ EM by 3.2 points, while disabling adaptive granularity reduces performance by 5.1 points on average.

The full ablation results are summarized in Table A1. Each row represents removing one component from the full AdaptRAG system, demonstrating the importance of each design decision.
</Page 8>
