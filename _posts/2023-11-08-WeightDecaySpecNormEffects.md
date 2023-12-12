---
layout: distill
title: Exploring Weight decay and Spectral Normalization in MLPs and Residual networks
description: Project proposal for Spectral normalization related final project for 6.s898, Fall 2023.
date: 2023-11-08
htmlwidgets: true

# Anonymize when submitting
# authors:
#   - name: Anonymous

authors:
  - name: Preston Hess
    url: "https://rphess.cargo.site/"
    affiliations:
      name: MIT BCS and EECS
  - name: Andrew Hutchison
    affiliations:
      name: MIT EECS

# must be the exact same name as your blogpost
bibliography: _biblography/2023-11-08-WeightDecaySpecNormEffects.bib  

# Add a table of contents to your post.
#   - make sure that TOC names match the actual section names
#     for hyperlinks within the post to work correctly.
toc:
  - name: Relevance and Investigation
  - name: Proposed Methods
---

## Relevance and Investigation

Weight normalization is important in machine learning for two reasons. Weight normalization prevents weights from getting too large, thereby avoiding exploding gradients and introducing numerical stability while training. Furthermore, it can prevent overfitting to the data. One popular method for weight normalization is weight decay. Weight decay is a regularization technique that penalizes the Frobenius Norm of the weight matrices. It is implemented through adding a term proportional to the sum of the Frobenius Norm of the weight matrices to the loss function, thereby increasing loss when weights get larger. One of the issues with merely regularizing with the Frobenius Norm or performing Frobenius normalization of weight matrices is that it imposes a more strict constraint than we want: it enforces that the sum of singular values is one, which can lead to weight matrices of rank one (Miyato et al. 2018).  Another issue is that the sum of the Frobenius norm scales with depth, potentially causing deeper networks to force smaller values than necessary upon their weight matrices. 

A more novel method that addresses this is spectral normalization, which instead focuses on initializing and updating the weight matrices in a way that preserves their spectral norm, keeping it around the square root of the change in layer size. This deals with some issues of weight decay by focusing on the norms of individual weight matrices during their update, rather than summing the effect of all weight matrices in the loss function. Thus far, it seems to allow for a more stable learning algorithm and helps to produce more predictable scaling of models and improve feature learning. 

We want to further explore the effects of weight decay and spectral normalization on different architectures through a comparative study on Multi-Layer Perceptrons (MLPs) and Residual Neural Networks (ResNets). We aim to investigate two general areas related to the spectral norm: spectral normalization versus Weight Decay, and differences in the influence of spectral normalization on MLPs and Residual Neural Networks. We aim to understand how the spectral norm of weight matrices change over time, how the rank of weight matrices is affected by each technique, and how they affect overall model performance. Furthermore, we want to see how the distribution of singular values changes across architectures, determining if certain types of architectures can benefit more from spectral normalization than another. 

## Proposed Methods

We will train MLPs and ResNets of two depths- medium and large- on a simple image classification task. Within each of these 4 classes we will train each network with no weight normalization to act as a baseline, with weight decay, and with spectral normalization. During training we will keep track of the metrics of interest at the end of each epoch. We plan to train our models using Preston’s access to MIT BCS’s OpenMind compute cluster, where we will have access to extensive compute resources that should make training time trivial. 

Instead of only investigating the effects of our independent variables on accuracy, we will record the distribution of singular values across epochs and trials to see if we can find any important trends in terms of predicting performance. More importantly, this investigation will help illuminate any underlying mechanistic reasons for certain properties of our network. We will also record how the rank of weight matrices changes over time for different normalization methods and architectures. More discussion is needed with our advisor in order to understand the significance of low rank weight matrices and how we might incorporate this into our analysis.

***
