---
layout: distill
title: Exploring limited and noisy datasets augmentation using denoising VAEs
description: 
date: 2023-11-11
htmlwidgets: true

# Anonymize when submitting
# authors:
#   - name: Anonymous

authors:
  - name: Pranay Agrawal
    affiliations:
      name: MIT
  
# must be the exact same name as your blogpost
bibliography: 2023-11-11-denoisingVAE.bib  

# Add a table of contents to your post.
#   - make sure that TOC names match the actual section names
#     for hyperlinks within the post to work correctly.
toc:
  - name: Introduction
  - name: Objective
  - name: Research questions to explore

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

Denoising autoencoders (DAE) are trained to reconstruct their clean inputs with noise injected at the input level, while variational autoencoders (VAE) are trained with noise injected in their stochastic hidden layer, with a regularizer (KL divergence) that encourages this noise injection. 
Denoising Variational Autoencoders (DVAEs) are an extension of the traditional variational autoencoder (VAE). The research paper "Denoising Criterion for Variational Auto-Encoding Framework" <d-cite key="denoisingVAE"></d-cite> elucidates how incorporating a denoising criterion into the VAE framework can significantly improve the robustness of the learned representations, thereby enhancing the model's generalization ability over various tasks.

## Objective
The aim is - 
1. to develop a DVAE OR use a pre-trained model that is capable of extracting robust features from small and noisy datasets, such as the RETINA dataset for diabetic retinopathy diagnosis. 
2. test if generated synthetic data can supplement the original dataset, enhancing the performance in downstream tasks with scarce data/imbalanced classes.

  
## Research questions to explore

1. **Learning Robust representation and Generating Synthetic data using DVAEs:** Can DVAEs dual capability of denoising input data and learning a generative model of the data distribution simultaneously be exploited to effectively learn robust representations from limited and noisy datasets and utilized to generate additional synthetic data (augmented dataset)? 

2. **Performance Enhancement for downstream tasks:** How does the DVAE-generated synthetic data impact the performance metrics of downstream tasks, for example, severity classification?

3. **Comaprison with traditional VAEs:** How the learned representaion using DVAEs compare to traditional VAEs on the noisy data? Does the denoising aspect of DVAEs provide a tangible benefit over traditional VAEs in terms of improved accuracy? Is the DVAE-augmented data robust to variations in image quality, such as those caused by different imaging equipment in healthcare data?

***
