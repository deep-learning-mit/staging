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

Time-series data analysis is pivotal in numerous scientific and industrial applications, including dynamical system, weather forecasting, and stock market prediction. The underlying parameters governing the time-series data can often be complex and not directly observable. Unlike traditional time series approaches, which predominantly focus on prediction tasks, leading to a "black-box" prediction. In this project, we want to leverage the contrastive learning approach studied in class to learn latent parameters. 

A deep comprehension of these underlying parameters, if successfully achieved, can lead to 2 benefits - 1) enhanced model capability for making accurate future predictions, and 2) a better understanding of the underlying system. The latter is particularly important in scientific, where the goal is to understand the underlying system, and engineering, where safety and reliability are of paramount importance.

To achieve the above goals, we proposed the following experiments and setups to study the insights of using contrastive approach to learn latent parameters for time-series representation.


## Objectives
The primary objective of this research is to investigate the effectiveness of contrastive loss learning in capturing the system underlying parameters ($$\theta_i$$) of time-series data. We aim to:

1. Test the capability of contrastive learning approach to extract embeddings from time-series data that correlate strongly with system underlying parameters.
2. Study different neural network architecture for encoding time-series trajectories into informative embeddings.
3. Explore the impact of various factors such as function forms, number of parameters and distributions, trajectory length, noise levels, and loss functions on the model’s performance.
4. Evaluate the precision of the predictive models in terms of their ability to make accurate future predictions based on learned latent variables, particularly in few-shot learning scenarios.

## Hypothesis
With contrastive loss learning, the embeddings of trajectories from the same parameter set will be closer together in the embedding space than to those from different sets. Therefore, our central hypothesis is that the embeddings produced by a model trained with contrastive loss learning will reflect the underlying parameters of time-series data. It is anticipated that a linear projection of these embeddings back onto the parameter space will yield predictions that are congruent with the original parameter values. Moreover, we postulate that the model will be able to make more precise future predictions by effectively capturing the essence of the latent variables governing the time-series data.

## Experimental Setup

### Trajectories Simulation

We will generate synthetic time-series data based on underlying deterministic and stochastic processes (e.g., spring-mass dynamical system). 
- The system can be defined by a set of parameters $$\theta_i$$. We have $H$ set of parameters.
- For each set of parameters, a trajectory, $$\{x_{ij}\}$$ of length $T$ can be draw with different initial conditions and noise. We will sample $K$ trajectories for each set of parameters.

### Models

We will evaluate three different neural network architectures:

1. Recurrent Neural Network (RNN)
2. Long Short-Term Memory (LSTM)
3. Transformer (utilizing attention mechanisms)

A model $$M$$ will output an embedding vector $$v_{ij}$$ for a given input trajectory $$\{x_{ij}\}$$.

### Experimentation

We want to evaluate the contrastive approach in extracting system parameter under the following scenarios:

1. **System Functional Forms:** We will test linear, non-linear, and complex periodic functions to generate the trajectories.
2. **Number of Parameters ($$\lvert \theta \rvert$$):** We will explore varying the number of parameters to understand how it affects the model’s ability to learn.
3. **Parameter Distribution:** We will use different distributions (uniform, normal, bimodal, etc.) of parameters (i.e., $\theta_i$) to study the impact on the learning process.
4. **Trajectory Length ($$T$$):** We will vary the length to assess the effect on the model’s performance.
5. **Noise Levels:** Different amounts of Gaussian noise will be added to the trajectories to simulate real-world data imperfections.
6. **Loss Functions:** Alongside contrastive loss, does add a loss function for model prediction of next time stamp help performance?


## Conclusion

This proposal presents a structured plan to investigate the potential of contrastive loss approach in learning system underlying parameters of time-series data. The insights gained from this research could pave the way for advancements in various fields where time-series analysis is crucial. We hope the insights from our project can contribute to the field of machine learning and its applications in time-series analysis.