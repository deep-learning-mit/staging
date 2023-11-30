---
layout: distill
title: Project Proposal
description: A survey of various embeddings for spatio-temporal forecasting.
date: 2023-11-08
htmlwidgets: true

# Anonymize when submitting
# authors:
#   - name: Anonymous

authors:
  - name: Joshua Sohn
    # url: "https://en.wikipedia.org/wiki/Albert_Einstein"
    affiliations:
      name: MIT
  - name: Samuel Lee
    # url: "https://en.wikipedia.org/wiki/Boris_Podolsky"
    affiliations:
      name: MIT

# must be the exact same name as your blogpost
bibliography: 2023-11-08-spatiotemporal.bib  

# Add a table of contents to your post.
#   - make sure that TOC names match the actual section names
#     for hyperlinks within the post to work correctly.
toc:
  - name: Abstract
  - name: Related Work
  - name: Methodology
  - name: Evaluation

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

## Abstract

Time series forecasting is an interdisciplinary field that affects various domains, including finance and healthcare, where autoregressive modeling is used for informed decision-making. While many forecasting techniques focus solely on the temporal or spatial relationships within the input data, we have found that few use both. Our goal is to compare robust embeddings that capture both the spatial and temporal information inherent in datasets and possibly devise one ourselves. We will focus on the field of traffic congestion, which is a pervasive challenge in urban areas, leading to wasted time, increased fuel consumption, and environmental pollution. Accurate traffic flow forecasting is critical for traffic management, infrastructure planning, and the development of intelligent transportation systems. Through this project, we hope to discover the most effective method of generating spatiotemporal embeddings in traffic flow forecasting models. 

## Related Work

Currently, there are three different embedding techniques that we will be comparing in our project. 

The first is the Spatio-Temporal Adaptive Embedding transformer (STAEformer)<d-cite key="liu2023staeformer"></d-cite>.
STAEformer uses adaptive embeddings, which adds an embedding layer on the input to dynamically generate learned embeddings on the dataset. In their architecture, the input embedding is then fed into temporal and spatial transformer layers, followed by a regression layer. 

{% include figure.html path="assets/img/2023-11-08-spatiotemporal/staeformer_architecture.png" class="img-fluid" %}
<div class="caption">
    Architecture of the Spatio-Temporal Adaptive Embedding transformer (STAEformer).<d-cite key="liu2023staeformer"></d-cite>
</div>

The second is the Spatio-Temporal Transformer with Relative Embeddings (STTRE)<d-cite key="deihim2023sttre"></d-cite>. STTRE uses relative position encodings, renamed as relative embeddings. The idea to leverage relative embeddings as a way to capture the spatial and temporal dependencies in the dataset of a multivariate time series. In their architecture, the relative embeddings are coupled with a transformer with multi-headed attention. 

The third is the Spacetimeformer<d-cite key="grigsby2023spacetimeformer"></d-cite>. Spacetimeformer uses embeddings generated from breaking down standard embeddings into elongated spatiotemporal sequences. In their architecture, these embeddings are fed into a variant of the transformer model using local, global, and cross self-attention.

As the project progresses, we will continue looking for novel embeddings that have reached or are close to the sota benchmark in spatiotemporal forecasting and apply them to our model.

## Methodology
In order to investigate the most effective method of generating spatiotemporal embeddings, we will standardize the rest of the architecture. After our embedding layer, we will build our own transformer model with a single spatiotemporal layer. This will be followed by a regression layer that outputs the prediction. We will keep these parts relatively simple to focus on the embedding layer, which is where we’ll incorporate the different techniques described in the related works section. We will also perform some ablation experiments to measure the efficacy of the methods used to generate the spatiotemporal embeddings 

To train and test our model, we will use traffic forecasting datasets that are available online. We are considering using the METR-LA dataset<d-cite key="metr-la"></d-cite> and the PEMS-BAY dataset<d-cite key="pems-bay"></d-cite> as they are popular choices in this field.

If creating our own model seems infeasible, we will take an existing model and focus solely on the embedding layer. We’re currently settling on the STAEformer, as it outperformed the Spacetimeformer on the PEMS-BAY dataset when compared using the same performance metrics.

## Evaluation
We will be using common evaluation metrics in forecasting, such as MAE, MAPE, and MSE. We will also include the final accuracy of our model on the METR-LA and PEMS-BAY datasets.
