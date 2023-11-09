---
layout: distill
title: Contrastive Time Series Representation Learning
description: Proposal for a new method of time series representation learning
date: 2022-11-08
htmlwidgets: true


# Anonymize when submitting
# authors:
#   - name: Anonymous

authors:
  - name: Martin Ma
    url: "https://www.linkedin.com/in/martinzwm/"
    affiliations:
      name: Harvard University
  - name: Lily Wang
    url: "https://www.linkedin.com/in/xiaochen-lily-wang-175897183/"
    affiliations:
      name: Harvard University

# must be the exact same name as your blogpost
bibliography: assets/bibliography/2023-11-08-contrastive-time.bib  

# Add a table of contents to your post.
#   - make sure that TOC names match the actual section names
#     for hyperlinks within the post to work correctly.
toc:
  - name: Introduction
  - name: Objectives
  - name: Hypothesis
  - name: Experimental Setup
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

Time-series data analysis is pivotal in numerous scientific and industrial applications, including weather forecasting, stock market prediction, and natural language processing. The underlying parameters governing the time-series data can often be complex and not directly observable. Unlike traditional time series approaches, which predominantly focus on direct prediction tasks, our work outlines an approach to learn these latent parameters by employing contrastive loss learning, a technique typically used in self-supervised learning for tasks such as image and speech recognition. We postulate that a deep comprehension of these underlying factors is pivotal, and if successfully achieved, it should naturally enhance the model's capability for making accurate future predictions. The proposal also sets forth a comprehensive experimental framework to explore how various setups affect model performance, thereby contributing valuable insights into the adaptability of contrastive loss learning in new domains.


## Objectives
The primary objective of this research is to investigate the effectiveness of contrastive loss learning in capturing the latent parameters ($$z_i$$) of time-series data. We aim to:

1. Test the capability of contrastive learning loss to create discriminative embeddings from time-series data that correlate strongly with latent parameters.
2. Determine the optimal neural network architecture for encoding time-series trajectories into informative embeddings.
3. Explore the impact of various factors such as function forms, parameter count and distributions, trajectory length, noise levels, and loss functions on the model’s performance.
4. Evaluate the precision of the predictive models in terms of their ability to make accurate future predictions based on learned latent variables, particularly in few-shot learning scenarios.

## Hypothesis
With contrastive loss learning, the embeddings of trajectories from the same parameter set will be closer together in the embedding space than to those from different sets. Therefore, our central hypothesis is that the embeddings produced by a model trained with contrastive loss learning will reflect the underlying parameters of time-series data. It is anticipated that a linear projection of these embeddings back onto the parameter space will yield predictions that are congruent with the original parameter values. Moreover, we postulate that the model will be able to make more precise future predictions by effectively capturing the essence of the latent variables governing the time-series data.

## Experimental Setup

### Trajectories Simulation

We will generate synthetic time-series data based on various deterministic and stochastic processes. The parameters $$z_i$$ will serve as inputs to these functions to create $$K$$ trajectories for each $$H$$ set of parameters.

### Models

We will evaluate three different neural network architectures:

1. Recurrent Neural Network (RNN)
2. Long Short-Term Memory (LSTM)
3. Transformer (utilizing attention mechanisms)

Each model $$M$$ will output an embedding vector $$v$$ for a given input trajectory.

### Experimentation

Our experiments will involve the following variables:

1. **Function Forms:** We will test linear, non-linear, and complex periodic functions to generate the trajectories.
2. **Number of Parameters ($$n$$):** We will explore varying the number of parameters to understand how it affects the model’s ability to learn.
3. **Parameter Distribution:** We will use different distributions (uniform, normal, bimodal, etc.) to study the impact on the learning process.
4. **Trajectory Length ($$T$$):** We will vary the length to assess the effect on the model’s performance.
5. **Noise Levels:** Different amounts of Gaussian noise will be added to the trajectories to simulate real-world data imperfections.
6. **Loss Functions:** Alongside contrastive loss, we will also consider adding a mean squared error (MSE) loss for predicting the next value in a trajectory to see if it aids in learning.


## Conclusion

This proposal presents a structured plan to investigate the potential of contrastive loss learning in understanding and predicting the latent parameters of time-series data. The insights gained from this research could pave the way for advancements in various fields where time-series analysis is crucial. We are optimistic that the results will contribute significantly to the field of machine learning and its applications in time-series analysis.