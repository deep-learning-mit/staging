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
    subsections:
      - name: Inserting a Sparse Autoencoder in a Transformer
      - name: How We Train our Sparse Autoencoder
      - name: Fine-Tuning
  - name: Results  # Did we find that our SAE-surgery model performed under RLHF? Example: could it detect to minimize swear words?
    subsections:
      - name: Exploring a Sparse Autoencoder
      - name: Fine-Tuning with a Sparse Autoencoder
  - name: Discussion
  - name: Conclusion
  - name: Acknowledgements
---

## Introduction

Understanding how machine learning models arrive at the answers they do, known as *machine learning interpretability*, is becoming increasingly important as models are deployed more widely and in high-stakes scenarios. Without interpretability, models may exhibit bias, toxicity, hallucinations, dishonesty, or malice, without their users or their creators knowing. But machine learning models are notoriously difficult to interpret. Adding to the challenge, the most widely used method for aligning language models with human preferences, RLHF (Reinforcement Learning from Human Feedback), impacts model cognition in ways that researchers do not understand. In this work, inspired by recent advances in sparse autoencoders from Anthropic, we investigate how sparse autoencoders can help to interpret large language models. We contribute a novel, more interpretable form of fine-tuning that only learns parameters related to interpretable features of the sparse autoencoder.

{% include figure.html path="assets/img/2022-11-09-sparse-autoencoders-for-interpretable-rlhf.md/interpretability-hard-cartoon.png" class="img-fluid" %}
<div class="caption">
  Machine learning practitioners often cannot interpret the models they build (xkcd #1838).
</div>

## Related Work

Research on interpreting machine learning models falls broadly under one of two areas: representation-based interpretability (top-down) and mechanistic interpretability (bottom-up).

Representation-based interpretability seeks to map out meaningful directions in the representation space of models. For example, Li et al. found a direction in one model that causally corresponds to truthfulness <d-cite key="li2023inferencetime"></d-cite>. Subsequent work by Zou et al. borrows from neuroscience methods to find directions for hallucination, honesty, power, and morality, in addition to several others <d-cite key="zou2023representation"></d-cite>. But directions in representation space can prove brittle. As Marks et al. found, truthfulness directions for the same model can vary across datasets <d-cite key="marks2023geometry"></d-cite>. Moreover, current methods for extracting representation space directions largely rely on probing <d-cite key="belinkov2022probing"></d-cite> and the linearity hypothesis <d-cite key="elhage2022superposition"></d-cite>, but models may have an incentive to store some information in nonlinear ways. For example, Gurnee et al. showed that language models represent time and space using internal world models <d-cite key="gurnee2023language"></d-cite>; for a world model to store physical scales ranging from the size of the sun to the size of an electron, it may prefer a logarithmic representation.

Mechanistic interpretability, unlike representation engineering, studies individual neurons, layers, and circuits, seeking to map out model reasoning at a granular level. One challenge is that individual neurons often fire in response to many unrelated features, a phenomenon known as polysemanticity. For example, Olah et al. found polysemantic neurons in vision models, including one that fires on both cat legs and car fronts <d-cite key="olah2020zoom"></d-cite>. Olah et al. hypothesized that polysemanticity arises due to superposition, which is when the model attempts to learn more features than it has dimensions. Subsequent work investigated superposition in toy models, suggesting paths toward disentangling superposition in real models <d-cite key="elhage2022superposition"></d-cite>. Superposition is relevant for language models because the real world has billions of features that a model could learn (names, places, facts, etc.), while highly deployed models have many fewer hidden dimensions, such as 12,288 for GPT-3 <d-cite key="brown2020fewshot"></d-cite>.

Recently, Sharkey et al. proposed using sparse autoencoders to pull features out of superposition. In an interim research report, the team describes inserting a sparse autoencoder, which expands dimensionality, into the residual stream of a transformer layer <d-cite key="sharkey2022interim"></d-cite>. In a follow-up work, Cunningham et al. found that sparse autoencoders learn highly interpretable features in language models <d-cite key="cunningham2023sparse"></d-cite>. In a study on one-layer transformers, Anthropic provided further evidence that sparse autoencoders can tease interpretable features out of superposition <d-cite key="bricken2023monosemanticity"></d-cite>. Although interest in sparse autoencoders in machine learning is relatively recent, sparse autoencoders have been studied in neuroscience for many decades under the name of expansion recoding <d-cite key="albus1971cerebellar"></d-cite>.

Other researchers have begun to apply sparse autoencoders to other interpretability problems. For example, Marks et al. (different from previous Marks et al.) investigated whether models on which we perform RLHF internalize the reward signal. To do so, Marks compared sparse autoencoders trained on the base model with sparse autoencoders trained on the fine-tuned model <d-cite key="marks2023rlhf"></d-cite>. Our work is similar, but in an effort to better understand the dynamics of RLHF, we propose a new form of fine-tuning in which the learnable parameters are related to the interpretable features of the sparse autoencoder.

## Background

An **autoencoder** is an architecture for reproducing input data, with a dimensionality bottleneck. Let $d_\text{model}$ denote the dimension of the residual stream in a transformer (4096 for Pythia 6.9B). Let $d_\text{auto}$ denote the dimensionality of the autoencoder. To enforce the dimensionality bottleneck, we require $d_\text{model} > d_\text{auto}$. See the autoencoder diagram below:

{% include figure.html path="assets/img/2022-11-09-sparse-autoencoders-for-interpretable-rlhf.md/autoencoder.png" class="img-fluid" %}
<div class="caption">
    An autoencoder is trained to reproduce its input, subject to a dimensionality bottleneck.
</div>

A **sparse autoencoder** relies on a different kind of bottleneck, called sparsity. For a sparse autoencoder $g \circ f$ that acts on $x \in \mathbb{R}^{d_\text{model}}$ by sending $f(x) \in \mathbb{R}^{d_\text{auto}}$ and $g(f(x)) \in \mathbb{R}^{d_\text{model}}$, the training objective combines MSE loss with an $L^1$ sparsity penalty:

$$\mathcal{L}(x; f, g) = \|x - g(f(x))\|_2^2 + \beta \| f(x) \|_1,$$

where $\beta > 0$ trades off sparsity loss with reconstruction loss. With the sparsity constraint, we can now let $d_\text{auto} > d_\text{model}$ by a factor known as the *expansion factor*. In our work, we typically use an expansion factor of $4$ or $8$. The purpose of the sparse autoencoder is to expand out the dimension enough to overcome superposition. See the sparse autoencoder diagram below:

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

### How We Train our Sparse Autoencoder

We train our sparse autoencoder to reproduce MLP-post activations in layer one of Pythia 6.9B (deduplicated) <d-footnote>Deduplicated means that this Pythia 6.9B model was trained on scraped web text where duplicate articles and lengthy passages are removed. Because Pythia inherits from the GPT-NeoX architecture, the specific activations we collected are named gpt_neox.layers.1.mlp.dense_4h_to_h.</d-footnote>. To create a dataset of activations for training, we stream in text from [an open-source replication of WebText](https://huggingface.co/datasets/Skylion007/openwebtext), the dataset used to train GPT-2. For each batch of text, we collect Pythia 6.9B's MLP-post activations at layer one and use these activations as training data for the sparse autoencoder.

Concretely, our sparse autoencoder has four learnable parameters: $W_\text{enc}$, $W_\text{dec}$, $b_\text{enc}$, and $b_\text{dec}$. The second bias $b_\text{dec}$ is used to center the input. The sparse autoencoder encodes, applies a nonlinearity, and decodes its input $x$ as follows:

$$\text{SAE}(x) = \text{ReLU}((x - b_\text{dec})  W_\text{enc} + b_\text{enc}) W_\text{dec} + b_\text{dec}.$$

We constrain the rows of $W_\text{dec}$ to have unit norm by renormalizing after each optimizer step. Another approach to constrain the rows is to remove gradient information parallel to the feature vectors before each optimizer step, and also renormalize the rows. Although we did not implement it, Anthropic found that that the second approach [slightly reduces loss](https://transformer-circuits.pub/2023/monosemantic-features/index.html#appendix-autoencoder-optimization) <d-cite key="bricken2023monosemanticity"></d-cite>.

We use an expansion factor of $4$, meaning $d_\text{auto} = 16384$. When training, we use batch size $8$, learning rate $10^{-4}$, and default $\beta_1 = 0.9, \beta_2 = 0.999$ for the Adam optimizer. Because Pythia 6.9B's context length is $128$ tokens, each training step includes activations from $1024$ tokens. We save checkpoints every $20000$ steps ($20.48$ million tokens).

One subtlety in training is that the sparsity constraint can eventually cause some autoencoder neurons to never activate. How to best handle these so-called dead neurons is an open question. We follow Anthropic in [resampling dead neurons](https://transformer-circuits.pub/2023/monosemantic-features/index.html#appendix-autoencoder-resampling) to new values <d-cite key="bricken2023monosemanticity"></d-cite>. Because resampling can cause instability during training, we resample only every 10000 training steps. At that point, we say a sparse autoencoder neuron is dead if it has not activated in any of the last 5000 training steps. In an attempt to improve autoencoder performance, Anthropic resampled dead neurons to the feature directions in which the sparse autoencoder performed worst. For simplicity, we resample dead neurons by setting their corresponding rows of $W_\text{enc}$ and $W_\text{dec}$ to Kaiming uniform random vectors. We reset dead biases to zero.

### Fine-Tuning

We fine-tune Pythia 70M <d-footnote>We wanted to fine-tune Pythia 6.9B, but we encountered out-of-memory errors on an A100 GPU. In follow-up work, we will investigate quantization so that we can study Pythia 6.9B, including the sparse autoencoder we trained for it.</d-footnote> with our sparse autoencoder inserted in layer one <d-footnote>To learn the most about how fine-tuning affects transformer features, we would ideally learn interpretable feature directions at every transformer layer using a sparse autoencoder. Then, after fine-tuning, we could perform rich comparisons across the model. Unfortunately, reconstruction loss compounds across layers. With current training methods, it is only feasible for us to insert a sparse autoencoder into one layer of the transformer before performance significantly degrades.</d-footnote>. Instead of adjusting weights everywhere in the network, we constrain fine-tuning to adjust only a small set of interpretable parameters within the sparse autoencoder. In particular, we learn two vectors of dimension $d_\text{auto}$: a coefficient vector $c$ and a bias vector $d$. Just prior to applying $\text{ReLU}$, we scale the autoencoder hidden layer by $c$ and translate it by $d$.



We use a dataset of [FILL IN THE BLANK]. We train for [X] steps using the default Adam optimizer parameters and learning rate [X].

## Results

Our results come in two parts: an exploration of our trained sparse autoencoder on Pythia 6.9B and an analysis of fine-tuning using a smaller sparse autoencoder on Pythia 70M.

### Exploring a Sparse Autoencoder

When inserted into Pythia 6.9B at layer one, our sparse autoencoder achieves a loss of $3.201$ (zero-ablation degrades loss to $3.227) on the held-out dataset [WikiText-103](https://paperswithcode.com/dataset/wikitext-103), consisting of over 100M tokens from Good and Featured articles on Wikipedia. Pythia 6.9B's baseline loss is $3.193$. Notably, the sparse autoencoder outperforms a zero-ablation of the layer, demonstrating that it learned features that are useful for reconstruction.

As expected, if the sparse autoencoder is inserted into a layer it was not trained for, performance collapses. For example, if inserted at layer 31 of Pythia 6.9B, the loss becomes $12.586$. Below is a figure showing the additional loss from inserting the sparse autoencoder at the first eight layers of Pythia 6.9B.

{% include figure.html path="assets/img/2022-11-09-sparse-autoencoders-for-interpretable-rlhf.md/sae-losses.png" class="img-fluid" %}
<div class="caption">
  The sparse autoencoder preserves model performance in layer 1, the layer it was trained for. The green bar is loss on WikiText-103 of Pythia 6.9B on 5 random batches. The red bar is the additional loss incurred if the sparse autoencoder is inserted after the MLP at a given layer. The first eight layers are shown.
</div>

[[[FIND AT LEAST ONE INTERPRETABLE FEATURE AND INSERT SOME PICTURES SHOWING OUR ANALYSIS OF ITS INTERPRETATION]]]

### Fine-Tuning with a Sparse Autoencoder

[[[DESCRIBE RESULTS OF FINE-TUNING PYTHIA 70M.]]]

## Discussion

Our results on fine-tuning show that most sparse autoencoder features remain similar after fine-tuning. If the 

## Conclusion

Our work indicates that sparse autoencoders are a promising tool for machine learning interpretability. By inserting sparse autoencoders into transformer language models, we investigate how a novel form of fine-tuning can provide insight into changes in model behavior after fine-tuning. Although our current work is limited because we only fine-tune Pythia 70M, future work can scale up model size, compute resources, and the number of tokens used to train the sparse autoencoder. Additionally, future work can extend from investigating direct fine-tuning to investigating the effects of RLHF performed with Proximal Policy Optimization (PPO).

## Acknowledgements

We would like to thank Professor Isola, Professor Beery, and Dr. Bernstein for an introduction to fundamental perspectives in deep learning that will stay with us forever. Thank you to Logan Smith for invaluable early guidance on the questions we could explore related to sparse autoencoders. We are thankful for the AI Safety Student Team at Harvard (AISST) and MIT AI Alignment (MAIA) for lively office spaces and a supportive community.