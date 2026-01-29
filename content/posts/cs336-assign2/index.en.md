---
title: "[Stanford CS336] Assignment 5: Alignment and Reasoning Reinforcement Learning"
subtitle: ""
date: 2026-01-13T18:12:00+08:00
lastmod: 2026-01-13T18:13:00+08:00
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

## 1 Assignment Overview

In this assignment, you will gain hands-on experience in training language models to reason when solving math problems.

### What to Implement

1. Implement a zero-shot prompting baseline for the MATH competition dataset proposed by Hendrycks et al. [2021].
2. Supervised Fine-Tuning (SFT) using reasoning traces from a stronger reasoning model (DeepSeek R1, DeepSeekAI et al., 2025).
3. Use Expert Iteration to improve reasoning performance through verification rewards.
4. Use Group Relative Policy Optimization (GRPO) to improve reasoning performance through verification rewards.

For interested students, we will release an optional part of the assignment in the coming days: aligning language models with human preferences.

### What to Run

1. Evaluate the zero-shot prompting performance of the Qwen 2.5 Math 1.5B model (as a baseline).
2. Supervised fine-tuning of Qwen 2.5 Math 1.5B using R1's reasoning traces.
3. Expert iteration training of Qwen 2.5 Math 1.5B using verification rewards.
4. GRPO training of Qwen 2.5 Math 1.5B using verification rewards.

Assignment link:
[Assignment5-alignment GitHub Repository](https://github.com/Kosthi/assignment5-alignment)

Next, I will share some details and insights from completing the assignment.

## 2 Reasoning Capabilities of Language Models

### 2.1 Motivation

One prominent application of language models is building general-purpose systems that can handle various natural language processing tasks. This assignment focuses on an emerging application: mathematical reasoning. We will use this as a testbed to build evaluation systems, perform supervised fine-tuning, and explore methods for training language models to reason using reinforcement learning (RL).

This assignment differs from previous ones in two ways:

1. We no longer use the language model codebase and models from previous assignments. Ideally, we would use the base language models trained in previous assignments, but fine-tuning those models won't yield satisfactory results—they are too weak to demonstrate complex mathematical reasoning. Therefore, we switch to an accessible, modern, high-performance language model (Qwen 2.5 Math 1.5B Base) for most of our work.
2. We introduce a new benchmark dataset to evaluate language models. Previously, we considered cross-entropy as a good proxy for many downstream tasks. However, the core of this assignment is narrowing the gap between base models and downstream tasks, so we must use evaluation methods independent of cross-entropy. We will use the MATH 12K dataset proposed by Hendrycks et al. [2021], which contains challenging high school competition math problems. We evaluate language model performance by comparing model outputs against reference answers.

### 2.2 Chain-of-Thought Reasoning and Reasoning Reinforcement Learning

A recent hot trend in language models is using Chain-of-Thought (CoT) reasoning to improve performance on various tasks. Chain-of-thought refers to the process of reasoning through a problem step by step, generating intermediate reasoning steps before arriving at the final answer.

#### Chain-of-Thought Reasoning in Language Models

Early chain-of-thought methods decomposed problems into intermediate steps using a "scratchpad" and fine-tuned language models to solve simple math tasks like arithmetic [Nye et al., 2021]. Other research prompted strong models to "think step by step" before answering, finding this significantly improved performance on math reasoning tasks like elementary arithmetic [Wei et al., 2023].

#### Learning Reasoning via Expert Iteration

Self-Taught Reasoner (STaR) [Zelikman et al., 2022] constructs reasoning as a bootstrapping loop: the pretrained model first generates diverse chains of thought (CoTs), retaining only those that produce correct answers, then fine-tuning on these "expert" trajectories. Repeating this cycle improves the language model's reasoning ability and problem-solving rate. STaR demonstrated that this expert iteration approach [Anthony et al., 2017] can bootstrap reasoning capabilities through automatic string matching verification without human-written reasoning trajectories.

#### Reasoning Reinforcement Learning with Verification Rewards (o1, R1)

Recent research explores using more powerful reinforcement learning algorithms combined with verification rewards to improve reasoning performance. OpenAI's o1 (and subsequent o3/o4) [OpenAI et al., 2024], DeepSeek's R1 [DeepSeek-AI et al., 2025], and Moonshot's kimi k1.5 [Team et al., 2025] all use policy gradient methods [Sutton et al., 1999] to train on math and code tasks, verifying answer correctness through string matching or unit tests, achieving significant performance improvements on competition math and programming tasks. Follow-up work like Open-R1 [Face, 2025], SimpleRL-Zoo [Zeng et al., 2025], and TinyZero [Pan et al., 2025] demonstrate that pure reinforcement learning with verification rewards can improve reasoning performance even on models with only 1.5B parameters.

#### Experimental Setup: Models and Datasets

In the following sections, we will progressively adopt more complex methods to train base language models to solve math problems through step-by-step reasoning. This assignment uses the Qwen 2.5 Math 1.5B Base model, which is based on the Qwen 2.5 1.5B model and underwent continued pretraining on high-quality synthetic math pretraining data [Yang et al., 2024]. The MATH dataset is available at `/data/a5-alignment/MATH` on the Together cluster.

#### Open Source Alternative Datasets (For Open Source Auditors)

Due to copyright restrictions, the MATH dataset is not publicly available. If you're completing the assignment locally, you can use the following open-source math reasoning datasets:

- Countdown [Pan et al., 2025]: A simple synthetic task based on the UK TV show "Countdown", commonly used as a testbed for small-scale reasoning RL.
- GSM8K [Cobbe et al., 2021a]: Elementary arithmetic problem dataset, easier than MATH, suitable for debugging code correctness and familiarizing with reasoning RL workflow.
- Tulu 3 SFT Math [Lambert et al., 2025]: Synthetic math problems generated using GPT-4o and Claude 3.5 Sonnet. Since it's synthetic data, some answers (or even questions) may be incorrect.
- Other math SFT datasets available online.

If the dataset doesn't directly provide short ground truth labels (like 1/2), you can use math answer parsers like Math-Verify to process the ground truth label column.

## 3 Evaluating Zero-Shot MATH Performance

We first evaluate the base language model's performance on the MATH dataset's 5K test set. Establishing this baseline helps understand how subsequent methods affect model behavior.

### Problem (math_baseline_performance): 4 points

#### (a) Write a script to evaluate Qwen 2.5 Math 1.5B model's zero-shot performance on the MATH dataset

The script should complete the following tasks:

1. Load MATH validation samples from `/data/a5-alignment/MATH/validation.jsonl`;
2. Format these samples into string prompts acceptable by language models using the `r1_zero` prompt;
3. Generate model outputs for each sample;
4. Calculate evaluation metrics;
5. Serialize samples, model generations, and corresponding evaluation scores to disk for subsequent analysis.

During implementation, writing an `evaluate_vllm` method like below will be helpful, which you can reuse later:

```python
def evaluate_vllm(
    vllm_model: LLM,
    reward_fn: Callable[[str, str], dict[str, float]],
    prompts: List[str],
    eval_sampling_params: SamplingParams
) -> None:
    """
    Evaluate language model performance on a set of prompts,
    calculate evaluation metrics, and serialize results to disk.
    """
```

**Submission**: A script for evaluating zero-shot MATH dataset baseline performance.

See at [eval.py](https://github.com/Kosthi/assignment5-alignment/blob/main/eval.py)

#### (b) Run your evaluation script on the Qwen 2.5 Math 1.5B model

Count the number of model generations falling into each of the following three categories:

1. Both format reward and answer reward are 1 (completely correct);
2. Format reward is 1, answer reward is 0 (format correct, answer wrong);
3. Both format reward and answer reward are 0 (both format and answer wrong).

Observe at least 10 cases with format reward 0. Do you think the problem lies with the base model's output or the parser? Why? Then observe at least 10 cases with format reward 1 but answer reward 0. What are your thoughts?

**Submission**: Analysis of model and reward function performance, including examples of each category.

**1. Count Statistics**
Based on the summary at the beginning of the log (`num_examples=5000`) and category counts, the model generation distribution is:

- **Both format reward and answer reward are 1 (completely correct)**: `count(fmt=1,ans=1)=139` cases.
- **Format reward is 1, answer reward is 0 (format correct, answer wrong)**: `count(fmt=1,ans=0)=693` cases.
- **Both format reward and answer reward are 0 (both wrong)**: `count(fmt=0,ans=0)=4168` cases.

**2. Observations on "Format Reward 0" Cases**
In at least 10 such cases from the log (marked as `sample_fmt0_ans0_examples=10`), **the problem mainly lies with the base model's output**, not the parser.

**Main Reason**: The common issue in these failure cases is that the model did not follow the specified output format. According to the log, the correct format should include clear `<think>` reasoning and a final answer wrapped in `<answer>` tags. However, problematic model outputs often show:

- **Confused format structure**: For example, when answering "What is the positive difference between $120%$ of 30 and $130%$ of 20?", the model gave the answer directly within `<think>` tags and mixed in undefined tags like `<end think> <answer>`, causing parsing failure.
- **Irrelevant content and symbols**: For example, when answering the inverse function problem "Let $f(x)=7x+5$...", the `<think>` content contained lots of irrelevant code snippets, garbled characters, and even image links, completely deviating from mathematical reasoning.
- **Reasoning and answer not separated**: The model often wrote "the answer is..." in the thinking part, or the final answer wasn't wrapped in required tags.

These phenomena indicate that the base model has difficulty following strict structured output instructions and fails to generate text that parsers can correctly process.

**3. Observations on "Format Reward 1 but Answer Reward 0" Cases**
In at least 10 such cases from the log (marked as `sample_fmt1_ans0_examples=10`), the model successfully followed format requirements, but the answer itself was incorrect.

**Main Observations**: This reveals the model's **core capability limitations**. With correct formatting, error types include:

- **Mathematical calculation errors**: For example, calculating `$i^5+i^{-25}+i^{45}$`, the model got `$2i$`, but the correct answer should be `$i$`.
- **Logical reasoning errors**: For example, finding integers satisfying `$|x|+1>7$` and `$|x+1|\le7$`, the model correctly listed calculation steps but incorrectly included `-6`, getting sum `-21` when the correct answer is `-15`.
- **Imprecise final answer expression**: For example, asking about the number of vertical lines, the model correctly gave values causing denominator to be zero "`-3$ and $2$`" in `<answer>`, but didn't convert it to the final answer "2 lines".

This indicates that the model has somewhat learned to "imitate" the format and framework of problem-solving, but its mathematical reasoning, calculation accuracy, and understanding of what the problem ultimately requires remain insufficient.

#### (c) How does the Qwen 2.5 Math 1.5B model perform on zero-shot baseline on the MATH dataset?

**Submission**: Summarize evaluation metrics in 1-2 sentences.

Based on the log metrics (`mean_reward=0.0278`, and completely correct ratio only 139/5000≈2.78%), this model has **very weak zero-shot baseline performance** on the MATH dataset, with extremely low overall accuracy in both format and answer, not yet capable of reliably solving complex math problems.

## 4 Supervised Fine-Tuning for MATH

### Problem (sft_experiment): Run SFT on MATH dataset (2 points) (2 H100 hours)

##### 1. Using Qwen 2.5 Math 1.5B base model, run SFT on reasoning SFT examples (provided in /data/a5-alignment/MATH/sft.jsonl), varying the number of unique examples in SFT in {128, 256, 512, 1024}, and using the full dataset. Tune learning rate and batch size to achieve at least 15% validation accuracy when using the full dataset.

**Submission:** Validation accuracy curves associated with different dataset sizes.

Fewer SFT samples actually work better than more samples. Entropy also drops faster and lower. With more samples, the model becomes confused instead.

![sft_experiment-1](./sft_experiment-1.png)

##### 2. Filter reasoning SFT examples to include only those producing correct answers. Run SFT on the (full) filtered dataset and report the filtered dataset size and validation accuracy achieved.

**Submission:** Report dataset size and validation accuracy curve. Compare your findings with previous SFT experiments.

Experiment pending

## 5 Expert Iteration for MATH

### Problem (expert_iteration_experiment): Run Expert Iteration on MATH dataset (2 points) (6 H100 hours)

#### Run expert iteration on the MATH dataset (provided in /data/a5-alignment/MATH/train.jsonl) using Qwen 2.5 Math 1.5B Base model, varying the number of rollouts per question G and number of epochs used in SFT step, using n_ei_steps = 5. Vary batch size for each expert iteration step (i.e., size of Db) in {512, 1024, 2048}. (You don't need to try all possible combinations of these hyperparameters. Just enough to draw conclusions about each.) Record entropy of model responses during training. Make sure vLLM terminates generation at the second answer tag </answer> as done in SFT section.

Submissions:
- Validation accuracy curves associated with different rollout configurations. Try at least 2 different rollout counts and epoch counts.
- A model achieving at least 15% validation accuracy on MATH.
- A short 2-sentence discussion comparing your performance with SFT performance, as well as performance across EI steps.
- A plot showing entropy of model responses during training.

Experiment pending

## 6 Policy Gradient Primer
