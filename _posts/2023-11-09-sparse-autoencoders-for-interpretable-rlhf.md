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
  - name: Background  # quickly explaining autoencoder, sparse autoencoder (and any other concepts needed)
  - name: Methods # including one subsection rigorously defining our sparse autoencoders, one subsection for PPO and the reward model (including our auxiliary parameters for RLHF), and one subsection for the datasets we experimented on (openwebtext, and a little with chess)
  - name: Experiments  # Did we find that our SAE-surgery model performed under RLHF? Example: could it detect to minimize swear words?
  - name: Discussion
  - name: Future Directions
---

## Introduction

Understanding how machine learning models arrive at the answers they do, known as *interpretability*, is increasingly important as models become deployed in high-stakes scenarios. Otherwise models may exhibit bias, toxicity, hallucinations, or dishonesty without the user or the creators knowing. But machine learning models are notoriously difficult to interpret. Adding to the challenge, the most widely used method for aligning language models with human preferences, RLHF (Reinforcement Learning from Human Feedback), impacts model cognition in ways that researchers do not understand. In this work, inspired by recent advances in sparse autoencoders from Anthropic, we contribute a novel, more interpretable way to perform RLHF. Along the way, we introduce the building blocks, including sparse autoencoders and PPO (Proximal Policy Optimization). We assume familiarity with the transformer architecture.

{% include figure.html path="assets/img/2022-11-09-sparse-autoencoders-for-interpretable-rlhf.md/interpretability-hard-cartoon.png" class="img-fluid" %}

## Related Work

Research on interpreting machine learning models falls broadly under one of two areas: representation-based interpretability (top-down) and mechanistic interpretability (bottom-up).

Representation-based interpretability seeks to map out meaningful directions in the representation space of models. For example, Li et al. found a direction in one model that causally corresponds to truthfulness <d-cite key="li2023inferencetime"></d-cite>. Subsequent work by Zou et al. borrows from neuroscience methods to find directions for hallucination, honesty, power, and morality, in addition to several others <d-cite key="zou2023representation"></d-cite>. But directions in representation space can prove brittle. As Marks et al. find, truthfulness directions for the same model can vary across datasets <d-cite key="marks2023geometry"></d-cite>. Moreover, current methods for extracting representation space directions largely rely on probing <d-cite key="belinkov-2022-probing"></d-cite> and the linearity hypothesis <d-cite key="elhage2022superposition"></d-cite>, but models may have an incentive to store some information in nonlinear ways. For example, Gurnee et al. showed that language models represent time and space using internal world models <d-cite key="gurnee2023language"></d-cite>; for a world model to store physical scales ranging from the size of the sun to the size of an electron, it may prefer a logarithmic representation.

Mechanistic interpretability, unlike representation engineering, studies individual neurons, layers, and circuits, seeking to map out model reasoning at a granular level. One challenge is that individual neurons often fire in response to many unrelated features, a phenomenon known as polysemanticity. For example, Olah et al. found polysemantic neurons in vision models, including one that fires on both cat legs and car fronts <d-cite key="olah2020zoom"></d-cite>. Olah et al. hypothesize that polysemanticity arises due to superposition, which is when the model attempts to learn more features than it has dimensions. Subsequent work investigated superposition in toy models, suggesting paths toward disentangling superposition in real models <d-cite key="elhage2022superposition"></d-cite>. Superposition is relevant for language models because the real world has billions of features that a model could learn (names, places, facts, etc.), while highly deployed models have many fewer hidden dimensions, such as 12,288 for GPT-3 <d-cite key="brown2020fewshot"></d-cite>.

Recently, Sharkey et al. proposed using sparse autoencoders to pull features out of superposition. In an interim research report, the team describes inserting a sparse autoencoder, which expands dimensionality, into the residual stream of a transformer layer <d-cite key="sharkey2022interim"></d-cite>. In a follow-up work, Cunningham et al. find that sparse autoencoders learn highly interpretable features in language models <d-cite key="cunningham2023sparse"></d-cite>. In a study on one-layer transformers, Anthropic provided further evidence that sparse autoencoders can tease interpretable features out of superposition <d-cite key="bricken2023monosemanticity"></d-cite>. Although interest in sparse autoencoders in machine learning is relatively recent, sparse autoencoders have been studied in neuroscience for many decades under the name of expansion recoding <d-cite key="albus1971cerebellar"></d-cite>.

[Reinforcement learning paragraph] <d-cite key="marks2023rlhf"></d-cite>

## Background

An **autoencoder** is an architecture for reproducing input data, with a dimensionality bottleneck. Let $d_\text{model}$ denote the dimension of the residual stream in a transformer (4096 for Pythia 6.9B). Let $d_\text{auto}$ denote the dimensionality of the autoencoder. To enforce the dimensionality bottleneck, we require $d_\text{model} > d_\text{auto}$. See the autoencoder diagram below:

{% include figure.html path="assets/img/2022-11-09-sparse-autoencoders-for-interpretable-rlhf.md/autoencoder.png" class="img-fluid" %}
<div class="caption">
    An autoencoder is trained to reproduce its input, subject to a dimensionality bottleneck.
</div>

A **sparse autoencoder** relies on a different kind of bottleneck, called sparsity. For a sparse autoencoder $g \circ f$ that acts on $x \in \mathbb{R}^{d_\text{model}}$ by sending $f(x) \in \mathbb{R}^{d_\text{auto}}$ and $g(f(x)) \in \mathbb{R}^{d_\text{model}}$, the training objective combines MSE loss with an $L^1$ sparsity penalty:

$$\mathcal{L}(x; f, g) = \|x - g(f(x))\|_2^2 + \beta \| f(x) \|_1.$$

With a sparsity penalty, we can let $d_\text{auto} > d_\text{model}$ by a factor known as the *expansion factor*. In our work, we typically use an expansion factor of 4 or 8. The purpose of the sparse autoencoder is to expand out the dimension enough to overcome superposition. See the sparse autoencoder diagram below:

{% include figure.html path="assets/img/2022-11-09-sparse-autoencoders-for-interpretable-rlhf.md/sparse-autoencoder.png" class="img-fluid" %}
<div class="caption">
    A sparse autoencoder is trained to reproduce its input, subject to an $L^1$ sparsity bottleneck.
</div>

## Methods

Our main experiment is to insert a sparse autoencoder into a transformer layer, train the sparse autoencoder, and then use the fused model to perform a new, more interpretable form of fine-tuning. <d-footnote>While we originally planned to investigate RLHF, we determined that existing libraries could not perform PPO (Proximal Policy Optimization) on custom model architectures such as our transformer fused with a sparse autoencoder. As a result, we chose to investigate fine-tuning instead of RLHF.</d-footnote>

### Inserting a Sparse Autoencoder in a Transformer

There are three natural places to insert a sparse autoencoder into a transformer:

1. MLP activations before the nonlinearity
2. MLP activations before adding back to the residual stream
3. The residual stream directly

We choose the second option. The upside of operating in the MLP space is that MLP blocks may be in less superposition than the residual stream, given that MLPs may perform more isolated operations on residual stream subspaces. The upside of operating after the MLP projects down to the residual stream dimension is a matter of economy: because $d_\text{model} < d_\text{MLP}$, we can afford a larger expansion factor with the same memory resources.

{% include figure.html path="assets/img/2022-11-09-sparse-autoencoders-for-interpretable-rlhf.md/transformer-with-sae.png" class="img-fluid" %}
<div class="caption">
    We insert a sparse autoencoder into a transformer after the MLP, but before adding into the residual stream.
</div>

### Interactive Demo: Exploring a Sparse Autoencoder

Below is an interactive figure in which you can explore features learned by our sparse autoencoder. For each feature you investigate, you will see the top tokens that make the particular feature activate. In addition, you can see how the feature activates on new text that you enter.

```
Put demo here!
```

We manually interpreted our sparse autoencoder's features and produced hypotheses for when they fire.


(BELOW IS INCOMPLETE)

### Finding 


We choose MLP-post  The choice we hav Building on the highly interpretable feature directions learned by sparse autoencoders, we train sparse autoencoders with a focus on the Pythia 6.9B model <d-cite key="biderman2023pythia"></d-cite>. To investigate a more interpretable form of fine-tuning, we insert the trained sparse autoencoder into one MLP layer and restrict the fine-tuning to change only special weights associated with the sparse autoencoder. 

When inserting a sparse autoencoder into a transformer, one can choose where to put it out of a 


Specifically, we train our sparse autoencoder on layer one of Pythia 6.9B. 

$$yy$$

<div class="l-page">
  <iframe src="{{ 'assets/html/2023-11-09-sparse-autoencoders-for-interpretable-rlhf/plotly_demo_1.html' | relative_url }}" frameborder='0' scrolling='no' height="600px" width="100%"></iframe>
</div>


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
