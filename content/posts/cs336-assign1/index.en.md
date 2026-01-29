---
title: "[Stanford CS336] Assignment 1: Building a Transformer Language Model"
subtitle: ""
date: 2025-06-14T17:43:58+08:00
lastmod: 2025-06-14T17:43:58+08:00
draft: false
author: "Koschei"
authorLink: ""
description: ""
images: []
resources:
- name: ""
  src: ""

tags: ["CS336", "LLM"]
categories: ["LLM"]

lightgallery: true
---

## Why Should Systems Enthusiasts Learn Large Language Models?

In today's AI technology wave, mastering large model knowledge has become an essential skill for systems developers. By participating in **Stanford CS336 Large Model Systems Course**, I began my journey of building large models from scratch. This course is likely to become a landmark course in the systems field over the next 3 years (similar to the position of CMU 15-445 database course in recent years).

### Assignment Overview

This assignment implements a small language model through the following three modules:

1. **Tokenizer Design and Implementation** — Byte Pair Encoding (BPE) tokenizer
2. **Model Architecture Coding** — Transformer with Self-Attention mechanism
3. **Optimizer Development** — AdamW optimizer

> Assignment link: [Assignment1-Basics GitHub Repository](https://github.com/Kosthi/assignment1-basics)

Next, I will share some details and insights from completing the assignment.

---

## 1. Byte Pair Encoding (BPE) Tokenizer

### 1.1 Unicode Standard

**Problem (unicode1): Understanding Unicode (1 point)**

(a) What Unicode character does chr(0) return?

The NULL character, i.e., ASCII null character.

(b) How does the string representation ('__repr__()'') differ from its printed representation?

The repr() function displays the escape sequence, while printing displays nothing (empty character).

(c) What happens when this character appears in text?

Although the null character is invisible when printed, it still exists as part of the Python string. This shows that Python strings can contain invisible characters, and these null characters still affect string storage and processing.

### 1.2 Unicode Encoding

**Problem (unicode2): Unicode Encoding (1 point)**

(a) Why do we prefer training tokenizers on UTF-8 encoded bytes rather than UTF-16 or UTF-32?

When training tokenizers, we process byte sequences. UTF-8 represents common characters more compactly, reducing sequence length, which is more efficient for model training. Moreover, UTF-8 is backward compatible with ASCII, making it especially efficient for processing English text.

(b) Why is this decode function incorrect?

This function is incorrect because it decodes byte by byte, which fails for multi-byte UTF-8 characters.

(c) Give a two-byte sequence that cannot be decoded.

0x80 0x81 is invalid because in UTF-8, any byte starting with binary 10 must be a continuation byte, but here it appears as the first byte.

### 1.3 BPE Tokenizer Training Experiments

**BPE Optimization Strategies**

To pursue high performance, I used Pybind11 to bind C++ code: pre-tokenization is handled by Python, while the BPE merge process is delegated to C++. Main optimizations include:

1. **Parallelization** — Use OpenMP for parallel statistics, avoiding lock contention
2. **Lazy Deletion Queue** — Complexity reduced from O(NlogN) to O(KlogN)
3. **Incremental Updates** — Only update affected adjacent pairs
4. **Efficient Data Structures** — Integer IDs instead of strings, custom hash functions

**Performance Comparison** (TinyStoriesV2-GPT4-train dataset):

| Version           | BPE Merge Training | Speedup  |
| ----------------- | ------------------ | -------- |
| Python            | 10min++            | Baseline |
| C++ Unoptimized   | 366s               | ~2x      |
| C++ Optimized     | 1.08s              | **300x** |

> For detailed optimization principles and implementation, see: [High-Performance BPE Tokenizer Optimization: From 10 Minutes to 1 Second](/posts/bpe-optimization/)

**Problem (train_bpe_tinystories): Train BPE on TinyStories (2 points)**

(a) How long did training take, and how much memory did it use? What is the longest token in the vocabulary? Is it reasonable?

- Training time: 28.03s
- Memory: 10GB
- Longest token: " accomplishment", ID: 7159, length: 15 characters (including leading space)
- Reasonable: Yes

(b) Analyze code performance. Which part of tokenizer training takes the longest?

- **N**: Total distinct words
- **L**: Average word length
- **V**: Target vocabulary size
- **M**: Number of merges = V - 256 - |special_tokens|
- **K**: Occurrence count of specific pair
- **P**: Temporary pair frequency table

Before optimization, the BPE merge process takes the longest (6 minutes), with time complexity O(M × N × L) and space complexity O(N × L + P).

After optimization, it takes only about 1s, with time complexity O(N × L + M).

**Problem (train_bpe_expts_owt): Train BPE on OpenWebText (2 points)**

(a) Train a byte-level BPE tokenizer on OpenWebText with max vocabulary size 32,000. What is the longest token? Is it reasonable?

Longest tokens:
- ID: 25835 | Byte length: 64 | Content: 64 hyphens
- ID: 25821 | Byte length: 64 | Content: Repeated UTF-8 sequences

(b) Compare tokenizers trained on TinyStories vs OpenWebText.

Some tokens contain newline characters, which when written to file without escaping, split a single merge rule across multiple lines.

### 1.4 Tokenizer Experiments

**Problem (tokenizer_experiments): Tokenizer Experiments (4 points)**

(a) Sample 10 documents each from TinyStories and OpenWebText. What is the compression ratio (bytes/token) for each tokenizer?

- TinyStories-10K tokenizer on TinyStories: **4.14 bytes/token**
- OpenWebText-32K tokenizer on OpenWebText: **4.70 bytes/token**

(b) What happens when encoding OpenWebText with the TinyStories tokenizer?

Using the TinyStories-10K tokenizer on OpenWebText documents, compression ratio drops to **3.26 bytes/token**, indicating the smaller vocabulary (10K) produces more tokens for complex text.

(c) Estimate tokenizer throughput.

- TinyStories-10K: ~626,519.6 bytes/sec, encoding 825GB Pile takes ~16.4 days
- OpenWebText-32K: ~763,734.4 bytes/sec, takes ~13.4 days

(d) Why is uint16 an appropriate choice for token IDs?

Both vocabularies (10K and 32K) are less than 65,536 (2^16), so uint16 suffices, saving 50% storage vs uint32.

---

## 2. Transformer Resource Accounting

### 2.1 FLOPs Accounting Basics

Understanding the computational and memory footprint of Transformer components is useful. We will perform basic "FLOP accounting."

Most floating point operations in Transformers come from matrix multiplications, so the core idea is simple:

1. List all matrix multiplication operations in Transformer forward pass
2. Convert each to required FLOPs

> **Matrix Multiplication FLOPs Rule**: For matrices A ∈ ℝ^(m×n) and B ∈ ℝ^(n×p), the multiplication AB requires 2mnp FLOPs.

### 2.2 GPT-2 XL Resource Accounting

**Problem (transformer_accounting): Transformer LM Resource Accounting (5 points)**

**(a) GPT-2 XL Trainable Parameters**

GPT-2 XL configuration:
- vocab_size: 50,257
- context_length: 1,024
- num_layers: 48
- d_model: 1,600
- num_heads: 25
- d_ff: 6,400

**Token Embedding Layer**: 50,257 × 1,600 = 80,411,200 (~80M)

**Single Transformer Block**:
- MHA (Q/K/V + Output projections): 4 × 1,600 × 1,600 = 10,240,000 (~10M)
- FFN (SwiGLU with W1, W2, W3): 3 × 6,400 × 1,600 = 30,720,000 (~31M)
- 2 RMSNorm layers: 2 × 1,600 = 3,200

Total per layer: 40,963,200 (~41M)
48 layers total: 1,966,233,600 (~1.97B)

**Final RMSNorm**: 1,600 parameters

**Output Projection (LM Head)**: 50,257 × 1,600 = 80,411,200 (~80M)

**Total**: ~2.13 billion parameters, requiring ~8 GB memory in float32.

**(b) List all matrix multiplications for forward pass FLOPs**

| Operation                    | Dimensions                                      |
| ---------------------------- | ----------------------------------------------- |
| MHA Q/K/V Projections        | (seq, d_model) × (d_model, d_model) × 3 per layer |
| Attention Score Computation  | (seq, d_k) × (d_k, seq) per head per layer       |
| Attention Weighted Sum       | (seq, seq) × (seq, d_v) per head per layer       |
| MHA Output Projection        | (seq, d_model) × (d_model, d_model) per layer    |
| FFN Layers (W1, W2, W3)      | (seq, d_model) × (d_model, d_ff) × 3 per layer   |
| Output Layer (LM Head)       | (seq, d_model) × (d_model, vocab_size)           |

Total for seq_len = 1024: **~4.20 TFLOPs**

**(c) Which parts consume most FLOPs?**

- Attention Q/K/V projections: 17.96%
- Attention score/weighted sum: 0.30%
- Attention output projection: 5.99%
- FFN (all three layers): 71.83%
- Output layer: 3.92%

**FFN is the dominant component** (~72% of FLOPs).

**(d) How do FLOPs distributions change with model scale?**

| Component                  | GPT-2 Small | GPT-2 Medium | GPT-2 Large | GPT-2 XL |
| -------------------------- | ----------- | ------------ | ----------- | -------- |
| MHA Q/K/V Projections      | 13.84%      | 16.51%       | 17.47%      | 17.96%   |
| Attention Score/Sum        | 1.02%       | 0.68%        | 0.46%       | 0.30%    |
| FFN (all layers)           | 55.36%      | 66.04%       | 69.89%      | 71.83%   |
| Output Layer               | 25.16%      | 11.25%       | 6.35%       | 3.92%    |

**Key trends**:
- FFN proportion increases significantly with scale
- Attention score computation proportion decreases sharply
- Output layer proportion drops dramatically

**(e) What happens if context_length increases to 16,384?**

- Total FLOPs increase from ~3.5T to ~70.4T (~20x increase, not linear due to O(S²) attention)
- Attention module proportion increases from ~24% to ~28%
- FFN proportion decreases from ~72% to ~69%

---

## 3. Training Transformer Language Models

### 3.1 Cross-Entropy Loss

The Transformer LM defines distribution p_θ(x_{i+1} | x_{1:i}) for sequence positions.

Cross-entropy (negative log-likelihood) loss:

85302
\ell(	heta; D) = rac{1}{|D|} \sum_{x \in D} \sum_{i=1}^{m} -\log p_{	heta}(x_{i+1} | x_{1:i})
85302

**Problem (cross_entropy): Implement Cross-Entropy**

Implemented using numerically stable log_softmax.

### 3.2 Learning Rate Tuning

**Problem (learning_rate_tuning): Tune Learning Rate**

Testing SGD with different learning rates for 10 iterations:
- LR = 10: Loss slowly decays
- LR = 100: Loss rapidly decays, converging near zero
- LR = 1000: Loss explodes, clearly diverging

### 3.3 Implementing AdamW

**Problem (adamw): Implement AdamW**

AdamW implementation with:
- First moment estimate (m)
- Second moment estimate (v)
- Bias correction
- Decoupled weight decay

**Problem (AdamW Accounting): AdamW Training Resource Accounting**

**(a) Peak Memory for AdamW**

- Parameters: 4(2Vd + L(16d² + 2d) + d) bytes
- Gradients: Same as parameters
- Optimizer state (m and v): 2× parameter memory
- Activations: 4[L(16BTd + 2BhT²) + BTd + 2BTV] bytes

**(b) For GPT-2 XL, what is max batch size in 80GB?**

Total memory ≈ 14.45B + 31.70 GB

Maximum batch size: **3**

**(c) FLOPs for one AdamW step?**

~10 FLOPs per parameter for:
1. First moment update: 2 FLOPs
2. Second moment update: 2 FLOPs  
3. Bias correction: 2 FLOPs
4. Parameter update: 4 FLOPs

Total: 10 × 2.13B = ~21.3 GFLOPs per step

**(d) Training time for GPT-2 XL on single A100?**

- FLOPs per step: 6 × B × T × P ≈ 1.34 × 10^16
- Total FLOPs for 400K steps: ~5.35 × 10^21
- A100 at 50% MFU: 9.75 × 10^12 FLOP/s
- Training time: **~17.4 years** on single A100
