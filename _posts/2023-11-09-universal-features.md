---
layout: distill
title: Project Proposal
description: This project aims to study the universality of features in LLMs by studying sparse autoencoders trained on similar layers of different models. 

date: 2023-11-09
htmlwidgets: true

# Anonymize when submitting
# authors:
#   - name: Anonymous

authors:
  - name: Misha Gerovitch
    url: "https://www.linkedin.com/in/michael-gerovitch-2010a61b0/"
    affiliations:
      name: MIT
  - name: Asher Parker-Sartori
    affiliations:
      name: MIT

# must be the exact same name as your blogpost
bibliography: 2023-11-09-universal-features.bib

# Add a table of contents to your post.
#   - make sure that TOC names match the actual section names
#     for hyperlinks within the post to work correctly.
toc:
  - name: Introduction
  - name: Setup
  - name: Experiments
    subsections:
    - name: Same models, early layer
    - name: Same models, additional experiments
    - name: Different models
    - name: Model stitching
    - name: Comparing representations
  - name: Acknowledgements

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

The internal components of LLMs are not well understood. One of the main barriers to understanding how LLMs represent information is the effect of polysemanticity, where a single neuron is activates for many different concepts (e.g. academic citations, English dialogue, HTTP requests, and Korean text), a result of a high-dimensional space of concepts being compressed into the space of a neural network (for transformers, this is in the residual stream or layers of an MLP.) Sparse autoencoders, a form of dictionary learning, help to linearly disentangle polysemantic neurons into individual features that are ideally more interpretable <d-cite key="cunningham2023sparse"></d-cite> <d-cite key="bricken2023monosemanticity"></d-cite>. We aim to train sparse autoencoders to identify similarities between layers of different models, for example the first layers of two trained models with identical architectures but different starting seeds.

Once we have the sparse autoencoders, we will compare the activation distributions on different inputs. If same-architecture models have similar performance on predicting training data, we expect that their activation distributions may be similar. We aim to study how well the features match up at different layers and between various models. We then can ask more complex question:
- Do (same architecture) models have similar feature representations at various layers?
- Do different architecture model have similar feature representations at various layers?
- What if the layers are different sizes but in the same model family? What if they are in different model families?
- Do models trained on different data have similar feature representations?
- How can we measure similarity between representations?
- Can we use this to improve model stiching techniques?

## Setup
We have started looking at [Hoagy Cunningham's codebase](https://github.com/HoagyC/sparse_coding) for training autoencoders that they used for their initial paper <d-cite key="cunningham2023sparse"></d-cite>.

[Neel Nanda also has some starter code](https://github.com/neelnanda-io/1L-Sparse-Autoencoder).

We are planning to try a range of different models from Pythia-160m to Llama2-7b (/-chat). We have relatively easy access to the models through the [TransformerLens library](https://neelnanda-io.github.io/TransformerLens/generated/model_properties_table.html), but are looking for other sources of models in case we need them.

We understand that training sparse autoencoders takes time and resources and are accounting for this taking us a good chunk of our time initially. We are connected with other groups, including Logan Riggs-Smith from the original sparse autoencoders paper, who have experience training the autoencoder. We are also considering sharing our learned representations between multiple groups working on this research to facilitate faster progress on projects that rely on trained autoencoders. This would allow us to focus more on running experiments and testing our additional hypotheses.

We have access to compute resources supported by MIT AI Alignment.

## Experiments
Here are a few possible experiments we could run:

### Same models, early layer
The most basic version of this experiment is to take an early residual stream layer of a transformer and train a sparse autoencoder on it for two models that are exactly the same except for the starting seed. Afterwards, we run a bunch of inputs through the autoencoder to get the activation distributions. Once we have the activation distrubitions, we can compare them (see "Comparing representations" section below for discussion.)

### Same models, additional experiments
- We can try looking layers of models trained with different data (but still have the same architecture)
- We can look at layers of RLHF-ed (chat) model vs the not fine-tuned model
- We can look at later layers of a model (e.g. in MLP)
- We can vary which model we do this on (e.g. Pythia vs Llama)

### Different models
A starting point here would be looking at models in the same family but have different parameter count. It is trickier to construct an experiment here since layers may be different sizes. The easiest test would be to find two layers that have the same size and compare the autoencoder-learned representations of those layers. Alternatively, we could investigate if more information is stored in a single layer of a smaller model than a larger model or if the information from one layer of a larger model is spread between two of smaller one.

### Model stitching
(1) Can we stitch together two model (with a trained weight matrix) right before a sparse autoencoder (that was pre-trained before stitching) that would allow us to extract useful features from the left-stitched model using the right-stitched sparse autoencoder?

(2) Can the representations somehow help us figure out where in the model is a good place to stitch two models to minimize the amount of training needed to get good performance? Can we understand what existing model stitching methods work well?

### Comparing representations
The simplest, and most desirable, comparison of representations would be finding the permuation matrix of one that most closely yields the other, thus finding a one to one feature mapping. However, this may not be possible. Another method would involve training a weight matrix between the autoencoders, perhaps with regularization that promotes sparsity. 

Model stitching can also be a method of comparing neural representations <d-cite key="Bansal2021stitching"></d-cite>.

## Acknowledgements

Special thanks to Sam Marks for suggesting the initial experiment ideas and to [MIT AI Alignment](https://www.mitalignment.org/) for providing connections with mentorship and compute resources.