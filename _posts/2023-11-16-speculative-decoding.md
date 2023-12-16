---
layout: distill
title: Accelerating large model inference with speculative decoding - 6.s898
description: An investigation into methods to speed up autoregressive inference through increased parallelization, specifically through speculative sampling and decoding.
date: 2023-11-16
htmlwidgets: true

# Anonymize when submitting
# authors:
#   - name: Anonymous

authors:
  - name: Dakota Goldberg
    url: "/#"
    affiliations:
      name: MIT

# must be the exact same name as your blogpost
bibliography: 2023-11-16-speculative-decoding.bib

# Add a table of contents to your post.
#   - make sure that TOC names match the actual section names
#     for hyperlinks within the post to work correctly.
toc:
  - name: Introduction
    subsections:
      - name: Inference in autoregressive models
      - name: Speculative execution in processors
      - name: Applying speculative execution to model inference
      - name: Hierarchical speculative decoding
  - name: Current Work
    subsections:
      - name: General setup
      - name: Sampling $p(x)$
      - name: The Algorithm
      - name: Evaluation
  - name: Hierarchical Speculative Decoding
  - name: Experiments
    subsections:
      - name: General set-up for experiments
      - name: How many orders of magnitude larger should $M_p$ be than $M_q$?
      - name: Set-up for hierarchical speculative decoding
  - name: Results
    subsections:
      - name: Calculating $c$ for each model pair
      - name: The general effect of speculative decoding
      - name: Acceptance rates and wall time given $M_p$ and $M_q$
      - name: Results of hierarchical speculative decoding
  - name: Conclusion

# Below is an example of injecting additional post-specific styles.
# This is used in the 'Layouts' section of this post.
# If you use this post as a template, delete this _styles block.
_styles: >
  .fake-img {
    background: #bbb;
    border: 1px solid rgba(0, 0, 0, 0.1);
    box-shadow: 0 0px 4px rgba(0, 0, 0, 0.1);
    margin-bottom: 12px;
  }
  .fake-img p {
    font-family: monospace;
    color: white;
    text-align: left;
    margin: 12px 0;
    text-align: center;
    font-size: 16px;
  }
---

## Introduction

### Inference in autoregressive models

Autoregressive models, particularly transformers and RNNs, play a crucial role in tasks involving sequential data processing, such as natural language processing and time series analysis. However, a significant limitation of these models is their slow inference speed. The primary bottleneck in these models is associated with memory reads and writes, rather than arithmetic computations. This is especially problematic in larger models with vast parameter spaces, where efficient memory management is critical to performance. Further, these models generate outputs sequentially, one token at a time, with each new token depending on all previously generated tokens. This inherent sequential dependency limits the model’s ability to parallelize the token generation process, leading to inference latency much greater than that of models capable of processing data in parallel. The challenge is to overcome this sequential bottleneck without compromising the model's ability to accurately capture dependencies in the data.

The central question this project addresses is whether it's possible to introduce parallelism into the inference process of autoregressive models. A more specific aspect of this problem is whether probabilities for multiple tokens can be computed simultaneously, rather than processing each token individually. This project aims to enhance methods that have been proposed for parallelizing the decoding process, focusing on solutions that draw inspiration from speculative execution in processors and other systems design strategies.

### Speculative execution in processors

Speculative execution is a technique used in CPU architecture to improve processing speed. Instead of waiting for sequential execution of instructions, processors predict which instructions are likely to be executed next and start processing them in advance. If the prediction is correct, this leads to a significant reduction in latency, as the processor has preemptively executed necessary instructions. If the prediction is incorrect, the processor discards the speculative results and reverts to the correct execution path. This method effectively utilizes CPU resources that would otherwise remain idle during the waiting period, thus optimizing the overall processing speed and reducing latency.

### Applying speculative execution to model inference

Inspired by speculative execution in processors, this project explores how similar principles can be applied to accelerate inference in large autoregressive models. The concept involves generating multiple potential outputs in parallel, using a smaller or draft model, and then evaluating these outputs with the larger target model. This mimics the speculative execution process where multiple paths are explored simultaneously, with the most promising path being selected as the final output. This approach, referred to as "speculative sampling" or "speculative decoding," aims to introduce a level of parallelism in the inference process, enabling faster generation of outputs without compromising the quality or accuracy of the model’s predictions.

### Hierarchical speculative decoding

In addition to implementing already proposed speculative decoding techniques, this project investigates a strategy that has the potential further speed up inference: hierarchical speculative decoding. This method aims to accelerate the smaller approximation model with an even smaller, faster model. While I experiment with two-layer (traditional) and three-layer hierarchies in this project, one could theoretically extend this idea to create an _n_ layer hierarchy, assuming sufficient memory. Although researchers developing speculative decoding algorithms and sampling methods have mentioned the potential viability of hierarchical speculative decoding, none have tried to implement it. Thus, this project aims to find an efficient implementation of the approach and determine if it actually further speeds up inference.

## Current Work

Multiple papers have presented novel speculative decoding algorithms, with the nuance typically in the way that sampling is performed. The two most-referenced papers in this space are DeepMind's Accelerating Large Language Model Decoding with Speculative Sampling (Chen et al.) [(paper)](https://arxiv.org/pdf/2302.01318.pdf) and Google Research's Fast Inference from Transformers via Speculative Decoding (Leviathan et al.) [(paper)](https://arxiv.org/pdf/2211.17192.pdf). This project draws its architecture from the latter, so we will more explore its approach in-depth and describe how its shortcomings motivated the experiments in this project.

### General setup

The approach presented in Fast Inference from Transformers via Speculative Decoding (Leviathan et al.) aims to accelerate inference from a target transformer-like model $M_p$. We present a distilled version of the speculative decoding set-up, algorithm, and evaluation here.

We start with two models:

1. $M_p$ (the target model)
2. $M_q$ (a smaller approximation model)

$p(x_{t}|x_{<t})$ describes the sampling of token $x_t$ given pretext $x_{<t}$, and we will refer to this as just $p(x)$. The shorthand applies for $q(x)$.

Our goal is to generate $\gamma \in \mathbb{Z}^{+}$ completions quickly with the approximation model, check that the probability of those generations are identical to the target model's (in parallel), and then reject and resample starting from the first "wrong" generation.

### Sampling $p(x)$

In order to sample $p(x)$, we will sample $x \sim q(x)$ instead.

1. If $q(x)\leq p(x)$, we keep $x$
2. Otherwise, we reject $x$ with a $1-\frac{p(x)}{q(x)}$ probability.
   - If we end up rejecting $x$, we resample $x\sim\text{norm}(\max(0, p(x)-q(x)))$.

Basically, we want $x\sim p(x)$ to be _at least_ as likely as $x \sim q(x)$. Following the steps above is equivalent to just sampling $x \sim q(x)$, and the paper provides a comprehensive proof of this in its appendix.

### The Algorithm

We use an implementation of the following algorithm from Leviathan et al. We start with some conditioning $prefix$ (our starting tokens) and generate between $1$ and $\gamma+1$ tokens at once.

{% include figure.html path="assets/img/2023-11-16-speculative-decoding/Algorithm1.png" class="img-fluid" %}

### Evaluation

To evaluate the effectiveness of this approach, we need to calculate the total wall time improvement of speculative decoding versus normal inference on the target model.

To make this evaluation more simple, assume we can run $\gamma + 1$ concurrent evaluations of $M_p$ in parallel. Now, we just need to get the cost of running $M_q$ (the approximation model).

Let $c$ = the cost coefficient, which is the ratio between the time for a single run of $M_q$ and a single run of $M_p$. $c$ will depend only on our hardware and software implementation details.

Now, we need some measure of how well $M_q$ approximates $M_p$.

Let $\beta$ be the _acceptance rate_.

- $\beta_{x<t}$ is the probability of accepting $x_{t}\sim q(x_{t}|x_{<t})$ by speculative sampling.
- Assume that the $\beta$s are i.i.d.

Let $\alpha=E(\beta)$. This gives us the average acceptance rate across many samples, which is a good measure of how well $M_q$ approximates $M_p$.

The expectation of the number of generated tokens is now a bounded geometric function of $\alpha$ (bounded by $\gamma$) :$$E(\text{# of generated tokens}) = \frac{1-\alpha^{\gamma + 1}}{1-\alpha}$$Given this relationship, we can derive the expected improvement factor for the total wall time (assuming longer generations):$$\frac{1-\alpha^{\gamma+1}}{(1-\alpha)(\gamma c+1)}$$
For the sake of conciseness, we leave the full proof to the paper, but the general sketch relies on the fact that each run of Algorithm 1 costs $Tc\gamma + T$ (where $T$ is the cost of running one step of $M_p$). We run $M_q$ $\gamma$ times and $M_p$ once, and each run of Algorithm 1 produces $\frac{1-\alpha^{\gamma + 1}}{1-\alpha}$ tokens. Since the cost of producing a single token with a standard algorithm is $T$, we get the above improvement.

## Hierarchical Speculative Decoding

How much faster can we make model inference by accelerating the approximation model with an even smaller, faster model? Let's look at the case where we have three models:

1. **$M_p$:** The target model
2. **$M_q$:** The first-level approximation model, used to approximate $M_p$.
3. **$M_r$:** The second-level, even smaller approximation model, used to approximate $M_q$.

With the introduction of $M_r$, we now need to consider additional parameters:

- **$\gamma_r$:** The number of concurrent evaluations that can be run using $M_r$.
- **$\beta_r$:** The acceptance rate for $M_r$, analogous to $\beta$ for $M_q$.
- **$\alpha_r = E(\beta_r)$:** The average acceptance rate for $M_r$, representing how well $M_r$ approximates $M_q$.

Now, $\beta$ for $M_q$ becomes a function of $\beta_r$, reflecting the hierarchical nature of this setup. The acceptance rate $\beta$ for $M_q$ now depends on how effectively $M_r$ approximates $M_q$, which in turn approximates $M_p$.

We can hypothesize that the effectiveness of $M_q$ in approximating $M_p$ might now be influenced by the performance of $M_r$. This could mean that $\beta$, and consequently $\alpha$, might be a function of $\alpha_r$.

The expectation of the number of generated tokens would now need to consider the hierarchical relationship. A new formula would be required to calculate this expectation, taking into account the performances of both $M_q$ and $M_r$.

Finally, the expected improvement factor for the total wall time would also need to be recalculated to reflect this hierarchical structure. This would involve integrating the costs and efficiencies of $M_r$ into our existing model, which so far only considered $M_q$ and $M_p$.

Whether or not this approach will actually speed up the model in practice is left to be determined experimentally.

## Experiments

I experimented on multiple transformer model families, most notably `facebook/opt-125m`, `facebook/opt-1.3b`, and `facebook/opt-13b`.

The primary research questions I investigated include:

1. How many orders of magnitude larger should $M_p$ be than $M_q$ to achieve the maximal improvement?
2. To what extent does hierarchical speculative decoding further speed up inference?

### General set-up for experiments

- For the standard (non-hierarchical) speculative decoding, I implemented the algorithm exactly as described above.
  - I used a gamma value of 4
- I used both top-k sampling and nucleus sampling, with `k=20` and `p=0.9` constant throughout all experiments.
- I typically prompted the models with `input_text = "Once upon a"` and generated 20 tokens.
- I used consistent sets of seeds (such as `torch.manual_seed(898)`) when running the same experiment across multiple model combinations for the sake of reproducibility and so that I could more easily compare results across models on shorter generation lengths.

### How many orders of magnitude larger should $M_p$ be than $M_q$?

- To investigate this, I calculated inference time (tokens per second) on each of the following (approximator, target) model pairs:
  - `facebook/opt-125m`, `facebook/opt-1.3b`
  - `facebook/opt-125m`, `facebook/opt-13b`
  - `facebook/opt-1.3b`, `facebook/opt-13b`

### Set-up for hierarchical speculative decoding

I experimented with a three-level hierarchical approach using

1. Small approximation model $M_r$: `facebook/opt-125m`
2. Approximation model $M_q$: `facebook/opt-1.3b`
3. Target model $M_p$: `facebook/opt-13b`

To add hierarchical decoding to the algorithm, I replaced the sampling of $M_q$, where we typically sample $x \sim q(x)$ with a sampling process that mirrors the sampling from the target model. So we sample from $x\sim r(x)$ instead, keep if it's at least as likely in $q(x)$, and reject proportional to the likelihood of the sample under either model, adjusting the distribution as before if we need to sample again. This made the theoretical implementation rather simple, as we could re-use a lot of the code. The implementation in practice was slightly more difficult than expected, however, as my implementation of the two-layer speculative decoding didn't permit direct functional composition, and I had to restructure the implementation a bit.

## Results

### Calculating $c$ for each model pair

(The larger model is used as the target model $M_p$)

|          | opt-125m | opt-1.3b | opt-13b |
| -------- | -------- | -------- | ------- |
| opt-125m | 1        | N/A      | N/A     |
| opt-1.3b | 0.015    | 1        | N/A     |
| opt-13b  | 0.022    | 0.015    | 1       |

This gives insight into the relative efficiencies of the models when performing assisted inference.

### The general effect of speculative decoding

Wall time improvements from speculative decoding have already been documented, so these results are not novel, but I include them here for further proof that the algorithm works and for comparison with other results.

| Target Model | Approximation Model | Tokens/Second |
| ------------ | ------------------- | ------------- |
| opt-13b      | None                | 0.047         |
| opt-13b      | opt-1.3b            | 0.087         |
| opt-13b      | opt-125m            | 0.057         |
| opt-1.3b     | None                | 0.336         |
| opt-1.3b     | opt-125m            | 1.05          |

In all cases, including an approximation model increases the model's token per second inference rate.

### Acceptance rates and wall time given $M_p$ and $M_q$

| Target Model | Approximator Model | Tokens/Second | Acceptance Rate |
| ------------ | ------------------ | ------------- | --------------- |
| opt-1.3b     | opt-125m           | 1.05          | 38%             |
| opt-13b      | opt-125m           | 0.057         | 15%             |
| opt-13b      | opt-1.3b           | 0.087         | 19%             |

These results help us answer the question: _How many orders of magnitude larger should $M_p$ be than $M_q$?_

One order of magnitude seems to yield higher acceptance rates, and the smaller models were obviously faster.

### Results of hierarchical speculative decoding

| Target Model | Approximation Model | Tokens/Second | Acceptance Rate |
| ------------ | ------------------- | ------------- | --------------- |
| opt-13b      | None                | 0.047         | N/A             |
| opt-13b      | opt-1.3b            | 0.087         | 19%             |
| opt-13b      | opt-125m            | 0.057         | 15%             |
| opt-13b      | opt-1.3b, opt-125m  | 0.030         | 17%, 33%        |

I found that running the three-layer hierarchical speculative decoding _did not_ speed up model inference, but I hypothesize that this is because of compute limitations. Running all three models on my computer given the parallelization requirements of the algorithm forced the program to map data to devices in a less-efficient way. I wasn't able to find smaller pre-trained models with which I could test this on my local machine, so a future experiment should either train custom smaller models for the sake of inference in this setting or use a device with greater memory capacity.

## Conclusion

This project explored the potential of speculative decoding, a technique inspired by speculative execution in processors, to accelerate inference in autoregressive models like transformers. Our exploration focused on implementing and extending existing methods of speculative decoding, particularly the ones proposed in the seminal works by Chen et al. and Leviathan et al., while also introducing early experiments with concept of hierarchical speculative decoding, which is to be further investigated.
