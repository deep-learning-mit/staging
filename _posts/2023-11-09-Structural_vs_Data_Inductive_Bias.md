---
layout: distill
title: Structural vs Data Inductive Bias
description: Class project proposal
date: 2023-11-08
htmlwidgets: true

# Anonymize when submitting
# authors: Tony Jiang, Gabriel Gallardo
#   - name: Anonymous

authors:
  - name: Gabriel Gallardo
    url: ""
    affiliations:
      name: MIT, Cambridge
  - name: Tony Jiang
    url: ""
    affiliations:
      name: MIT, Cambridge

# must be the exact same name as your blogpost
bibliography: 2023-11-09-Structural_vs_Data_Inductive_Bias.bib  

# Add a table of contents to your post.
#   - make sure that TOC names match the actual section names
#     for hyperlinks within the post to work correctly.
toc:
  - name: Motivation
  - name: Research Question
  - name: Methodology

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
## Motivation ##

The transformative impact of vision transformer (ViT) architectures in the realm of deep learning has been profound, with their applications swiftly extending from computer vision tasks, competing with traditional neural network architectures like convolutional neural networks (CNNs). Despite their success, the intricacies of how architectural variations within ViTs influence their performance under different data conditions remain largely uncharted. Unraveling these subtleties holds the promise of not only enhancing the efficiency and effectiveness of ViTs but also of offering a window into the broader question of structural inductive biases in deep learning models.  

The paper "Data-induced constraints versus model-induced structural inductive bias" [1]<d-cite key="reference1"></d-cite> presents a thorough analysis of the benefits of data augmentations on model performance, especially when facing out-of-distribution data. It quantifies the trade-off between augmented and real data and suggests that augmentations can sometimes exceed the value of more training data. This research is relevant to our project as it provides a comparative backdrop; while it explores data-induced constraints and the impact of data augmentation, our study aims to extend the understanding to the domain of model-induced inductive biases by examining the impact of architectural variations in vision transformers.  

ViT could be heavy data-hungry like stated in [2]<d-cite key="reference2"></d-cite>. Which gives us the opportunity to explore how we can change the structure of the architecture in order to achieve high performance even with a limited data set, comparing it with data augmentation presented in [1]<d-cite key="reference1"></d-cite>. 

 

## Research Question ##

This study seeks to dissect the following pivotal questions: How do specific architectural variations within vision transformer models affect their performance. Understand and quantify the tradeoff between the changes in the architecture and the amount of training data. Our hypothesis is that with some appropriate architectural changes, we would not need as much training data and still achieve the same result.   

  

## Methodology ## 

We will start with a standard Vision Transformer architecture as our baseline. From here, we will introduce variations to the architecture, specifically in the attention mechanisms. We want to test different types of attention layers (such as local, global, and sparse attention layer) and explore additional mechanism changes (such as attention augmentation, gating, etc.) [3]<d-cite key="reference3"></d-cite>. 

Each model will undergo training and evaluation on the Cipher-10 dataset. To appraise the models' performance, we will use measurement metrics including accuracy and training/inference time. The experimental design will encompass training with and without data augmentation to discern the impact of data variety on the architectural efficacy. 

 

## Reference ## 

[1] Data-induced constraints versus model-induced structural inductive bias (https://arxiv.org/pdf/2210.06441.pdf) 

[2] Training Vision Transformers with Only 2040 Images (https://arxiv.org/pdf/2201.10728.pdf) 

[3] Distilling Inductive Bias: Knowledge Distillation Beyond Model Compression (https://arxiv.org/ftp/arxiv/papers/2310/2310.00369.pdf) 


