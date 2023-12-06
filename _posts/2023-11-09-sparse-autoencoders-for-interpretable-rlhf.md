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
bibliography: 2023-11-09-sparse-autoencoders-for-interpretable-rlhf.bib

# Add a table of contents to your post.
#   - make sure that TOC names match the actual section names
#     for hyperlinks within the post to work correctly.
toc:
  - name: Introduction  # including an interactive demo of our sparse autoencoder for Pythia 6.9B
  - name: Related Work  # literature review of 8-10 highly relevant papers or blogs, with 1-3 sentences thoughtfully summarizing and connecting the idea in each one
  - name: Methods # including one subsection rigorously defining our sparse autoencoders, one subsection for PPO and the reward model (including our auxiliary parameters for RLHF), and one subsection for the datasets we experimented on (openwebtext, and a little with chess)
  - name: Experiments  # Did we find that our SAE-surgery model performed under RLHF? Example: could it detect to minimize swear words?
  - name: Discussion
  - name: Future Directions
---

## Introduction

Understanding how machine learning models arrive at the answers they do, known as *interpretability*, is increasingly important as models become deployed in high-stakes scenarios. Otherwise models may exhibit bias, toxicity, hallucinations, or dishonesty without the user or the creators knowing. But machine learning models are notoriously difficult to interpret. Adding to the challenge, the most widely used method for aligning language models with human preferences, RLHF (Reinforcement Learning from Human Feedback), impacts model cognition in ways that researchers do not understand. In this work, inspired by recent advances in sparse autoencoders from Anthropic, we contribute a novel, more interpretable way to perform RLHF. Along the way, we introduce the building blocks, including sparse autoencoders and PPO (Proximal Policy Optimization). We assume familiarity with the transformer architecture.

{% include figure.html path="assets/img/2022-11-09-sparse-autoencoders-for-interpretable-rlhf.md/interpretability-hard-cartoon.png" class="img-fluid" %}

## Related Work

Research on interpreting machine learning models falls broadly under one of two areas: representation engineering and mechanistic interpretability.

Representation engineering seeks to map out meaningful directions in the representation space of models. For example, Li et al. found a direction in one model that causally corresponds to truthfulness <d-cite key="li2023inferencetime" />. Subsequent work by Zou et al. borrows from neuroscience methods to find directions for hallucination, honesty, power, and morality, in addition to several others <d-cite key="zou2023representation" />. But directions in representation space can prove brittle. As Marks et al. find, truthfulness directions for the same model can vary across datasets <d-cite key="marks2023geometry" />. Moreover, current methods for extracting representation space directions largely rely on probing <d-cite key="belinkov-2022-probing" /> and the linearity hypothesis <d-cite key="elhage2022superposition" />. Gurnee et al. showed that language models represent time and space using internal world models <d-cite key="gurnee2023language" />; but world models may have an incentive to store certain information in nonlinear ways, such as logarithmically, for example, which would allow reasoning about the size of the sun and the size of electrons in a consistent way without floating point errors.

Mechanistic interpretability, unlike representation engineering, studies individual neurons, layers, and circuits, seeking to map out model reasoning at a granular level. One challenge is that individual neurons often fire in response to many unrelated features, a phenomenon known as polysemanticity. For example, Olah et al. found polysemantic neurons in vision models, including one that fires on both cat legs and car fronts <d-cite key="olah2020zoom"></d-cite>. Olah et al. hypothesize that polysemanticity arises due to superposition, which is when the model attempts to learn more features than it has dimensions. Subsequent work investigated superposition in toy models, suggesting paths toward disentangling superposition in real models <d-cite key="elhage2022superposition" />. Superposition is relevant for language models because the real world has billions of features that a model could learn (names, places, facts, etc.), while highly deployed models have many fewer hidden dimensions, such as 12,288 for GPT-3 <d-cite key="brown2020fewshot" />.

Recently, Sharkey et al. proposed using sparse autoencoders to pull features out of superposition. In an interim research report, the team describes inserting a sparse autoencoder, which expands dimensionality, into the residual stream of a transformer layer <d-cite key="sharkey2022interim" />. In a follow-up work, Cunningham et al. find that sparse autoencoders learn highly interpretable features in language models <d-cite key="cunningham2023sparse">. In a study on one-layer transformers, Anthropic provided further evidence that sparse autoencoders can tease interpretable features out of superposition <d-cite key="bricken2023monosemanticity" />.

[Reinforcement learning paragraph] <d-cite key="marks2023rlhf" />

## Methods

Building on the highly interpretable feature directions learned by sparse autoencoders, our work proposes that these feature directions could induce more interpretable learnable parameters in RLHF.

The main question we wish to answer is:

    Can sparse autoencoders be used to define a more interpretable RLHF?

To answer this main question, we may need to investigate several further questions:

1. What metrics accurately describe effective, interpretable RLHF?
2. How do we measure how good a sparse autoencoder is?
3. How do we train the best sparse autoencoders we can?

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

## Experiments

We have made significant progress on our research agenda already.
* We have learned how to **load, run, and save** large models such as Pythia 7B from the popular open-source hub Hugging Face.
* We have [**trained sparse autoencoders**](https://huggingface.co/naomi-laker/sparse-autoencoder/tree/main) on Pythia 70M and Pythia 7B. We learned lessons from initial mistakes, such as the need to resample dead neurons while training the sparse autoencoder.
* We have begun to study the relevant methods from **reinforcement learning**, such as PPO and RLHF, using materials available from [ARENA](https://arena-ch2-rl.streamlit.app/).

Our progress is tracked in real time on our [Notion page](https://invited-hacksaw-2fb.notion.site/Dictionary-Learning-Extension-1cd89e4193194bd39f500e2905e996b4).

In the next weeks, we will pursue these goals:
1. Learn how to perform RLHF on large models such as Pythia 7B.
2. Apply RLHF to sparse autoencoders we train on Pythia 7B.
3. Iterate on our methods. Research is a learning process!

## Discussion

## Future Directions
