---
layout: distill
title: Sparse Autoencoders for a More Interpretable RLHF
description: Extending Anthropic's recent monosemanticity results toward defining new learnable parameters for RLHF.
date: 2023-11-09
htmlwidgets: true

# Anonymize when submitting
# authors:
#   - name: Anonymous

authors:
  - name: Laker Newhouse
    url: "https://www.linkedin.com/in/lakernewhouse/"
    affiliations:
      name: MIT
  - name: Naomi Bashkansky
    url: "https://www.linkedin.com/in/naomibas/"
    affiliations:
      name: Harvard

# must be the exact same name as your blogpost
bibliography: 2023-11-06-sparse_autoencoders_for_interpretable_rlhf.bib  

# Add a table of contents to your post.
#   - make sure that TOC names match the actual section names
#     for hyperlinks within the post to work correctly.
toc:
  - name: Introduction
  - name: Our Research Questions
  - name: Study Outline (Methods, Analysis, Metrics)
  - name: Progress and Next Steps
---

## Introduction

Transformer-based large language models are increasingly deployed in high-stakes scenarios, but we have only rudimentary methods to predict when and how these models will fail. Mechanistic interpretability seeks to catch failure modes before they arise by reverse-engineering specific learned circuitry. While exciting work has been done on interpreting the [attention heads](https://transformer-circuits.pub/2021/framework/index.html) of models, the MLPs -- both the hidden layer, and the residual stream post-MLP -- have remained more elusive.

Individual neurons and the residual stream are often difficult to interpret because neurons are **polysemantic**. A polysemantic neuron is one that activates in response to multiple unrelated features, such as “cat” and “car,” or “this text is in Arabic” and “this text is about DNA.” Some researchers hypothesize that NNs learn a compression scheme known as **[superposition](https://transformer-circuits.pub/2022/toy_model/index.html)**, and that superposition gives rise to polysemanticity. Superposition occurs when there are more features embedded inside a layer than there are dimensions in that layer. Since each feature is represented as a direction in activation space, the features then form an overcomplete basis of the activation space. This overcomplete basis can still lead to excellent performance if the features are sparse -- e.g., most text is not in Arabic -- and if nonlinearities can smooth over interference between features.

But in the past year, a promising new idea was proposed to take features out of superposition: **sparse autoencoders** (SAEs). Sparse autoencoders were first proposed in a [blog post](https://www.lesswrong.com/posts/z6QQJbtpkEAX3Aojj/interim-research-report-taking-features-out-of-superposition) in December 2022 by Lee Sharkey. In September 2023, two groups published further work on SAEs: Anthropic ([Bricken et al.](https://transformer-circuits.pub/2023/monosemantic-features/)) and a group of independent researchers ([Cunningham et al.](https://arxiv.org/abs/2309.08600)). In an SAE, the goal is to learn a sparse representation in the latent dimension, such that each neuron represents an interpretable feature. SAEs are typically applied either to the residual stream or to the hidden layer of an MLP. The SAE trains on both L2 reconstruction loss and L1 sparsity in its hidden layer. The hidden dimension of the autoencoder is usually much larger than its input dimension, for instance by a factor of 8.

## Our Research Questions

The main question we wish to answer is:

    Can sparse autoencoders be used to define a more interpretable RLHF?

To answer this main question, we may need to investigate several further questions:

1. What metrics accurately describe effective, interpretable RLHF?
2. How do we measure how good a sparse autoencoder is?
3. How do we train the best sparse autoencoders we can?

## Study Outline (Methods, Analysis, Metrics)

To explore how sparse autoencoders can support a more interpretable RLHF, we will begin with the following initial experiment. Rather than fine-tuning all the transformer's weights in RLHF, we will experiment with fine-tuning *only a smaller subset of more interpretable parameters*.

Specifically, given a transformer with a sparse autoencoder reconstructing the MLP output at a given layer, our first proposed method is to define new learnable parameters for **interpretable RLHF** as the coefficients which scale the output feature vectors. For example, if the reward model punishes curse words, and there is a feature vector in the autoencoder corresponding to curse words, then that coefficient could be learned as strongly negative.

We have many **open-source resources** at our disposal.
* Independent researcher Neel Nanda has [*replicated*](https://github.com/neelnanda-io/1L-Sparse-Autoencoder) Anthropic’s recent monosemanticity paper, including scripts for analyzing sparse autoencoders.
* Logan Smith from EleutherAI has open-source code for [*training sparse autoencoders*](https://github.com/loganriggs/sparse_coding).
* The open-source Pythia 7B language model comes with a *pre-trained reward model* that we will use for our reinforcement learning experiments.
* For compute resources, we plan to use an *A100 GPU* available through Google Colab Pro+.

We expect to pursue multiple iterations of training autoencoders and applying them to RLHF. Reinforcement learning is hard to begin with, and it will be harder when limiting ourselves to the smaller space of interpretable parameters. We are prepared to research best-practices in both reinforcement learning and sparse autoencoder training.

Our **metrics for success** will be:
1. The reconstruction loss, sparsity, and interpretability of sparse autoencoders we train.
2. The loss of the reward model on predictions our model makes after interpretable RLHF, compared to the same loss using RLHF not constrained to interpretable parameters.
3. New relationships and intuitions we can articulate about the effect of sparsity on RLHF performance and accuracy, perhaps across different sparsity objectives from L1 loss.

Science is an iterative process. Creating new state-of-the-art methods for RLHF is not our goal. Rather, **our mission is a deeper understanding of the dynamics of RLHF in the context of sparse autoencoders**, along with releasing community-building, open-source contributions of clean, extendable, and useful training code to help future researchers at the intersection of reinforcement learning and sparse autoencoders.

## Progress and Next Steps

We have made significant progress on our research agenda already.
* We have learned how to **load, run, and save** large models such as Pythia 7B from the popular open-source hub Hugging Face.
* We have [**trained sparse autoencoders**](https://huggingface.co/naomi-laker/sparse-autoencoder/tree/main) on Pythia 70M and Pythia 7B. We learned lessons from initial mistakes, such as the need to resample dead neurons while training the sparse autoencoder.
* We have begun to study the relevant methods from **reinforcement learning**, such as PPO and RLHF, using materials available from [ARENA](https://arena-ch2-rl.streamlit.app/).

Our progress is tracked in real time on our [Notion page](https://invited-hacksaw-2fb.notion.site/Dictionary-Learning-Extension-1cd89e4193194bd39f500e2905e996b4).

In the next weeks, we will pursue these goals:
1. Learn how to perform RLHF on large models such as Pythia 7B.
2. Apply RLHF to sparse autoencoders we train on Pythia 7B.
3. Iterate on our methods. Research is a learning process!