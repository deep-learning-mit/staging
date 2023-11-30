---
layout: distill
title: Building an effcient foundational vision encoder for medical imaging via distillation
description: Proposal for 6.S898 course project, Fall 2023
date: 2023-11-08
htmlwidgets: true

# Anonymize when submitting
# authors:
#   - name: Anonymous

authors:
  - name: Ming Yang (Max) Lu
    url: ""
    affiliations:
      name: EECS, MIT

# must be the exact same name as your blogpost
bibliography: 2023-11-08-building-an-effcient-foundational-vision-encoder-for-medical-imaging-via-distillation.bib

# Add a table of contents to your post.
#   - make sure that TOC names match the actual section names
#     for hyperlinks within the post to work correctly.
toc:
  - name: Background
  - name: Anticipate challenges
  - name: Experiments
---

## Background

Self-supervised representation learning has emerged as a powerful framework for pretraining general purpose vision encoder backbones in computer vision. Representative works including SimCLR <d-cite key="simclr"></d-cite>, MoCo <d-cite key='moco'></d-cite>, DINO <d-cite key='dino'></d-cite>, MAE <d-cite key="mae"></d-cite> and more have demonstrated that self-supervised pretraining coupled with large and diverse unlabeled data, massive compute and high-capacity deep neural networks can learn strong representations that transfer to a wide range of downstream tasks with limited to no labeled data.

Self-supervised pretrained vision encoders are especially impactful for the field of computational pathology image analysis, where each single whole slide image (WSI) is enormous in size (up to 150,000px $$\times$$ 150,000px) and computational workflows, both supervised and unsupervised would typically require the WSI to first be chunked and embedded into a low-dimensional representation using a pretrained encoder in order to fit into GPU memory. However, for commonly used tile sizes (e.g. 256px $$\times$$ 256px), each WSI may consist of tens of thousands of image tiles, incurring high computational costs for running a large, pretrained encoder based on ViT-L or ViT-H on a single WSI, and making scaling to large datasets of tens of thousands of WSIs prohibitively expensive.

In this project, we propose to investigate the use of distillation to compress a large, pretrained foundational vision encoder for histopathology images into a smaller, more computationally and energy efficient one that can be used for downstream tasks in computational pathology while maintaining high performance. Using UNI <d-cite key="uni"></d-cite>, a recently developped foundational ViT-L model pretrained using DINO-V2 <d-cite key="dinov2"></d-cite> on 100M histopathology images as the teacher model, we will investigate multiple different setups for distillation such as distilling only the final global image representation vs. distillating the full dense feature map. For the student model, we will explore EfficientViT <d-cite key="efficientvit"></d-cite>, a recently proposed architecture that combines linear multiscale attention and depthwise separable convolution to obtain SOTA trade-off between accuracy and latency. We outline more details of our proposed experiments and anticipated challenges and findings below.

## Anticipated challenges

The anticipated challenges and some potential solutions are listed below:

- There is likely a mismatch between the semantic meaning of tokens in the teacher model (ViT) and student model (EfficientViT) due to difference in receptive field and size of feature maps, which may lead to suboptimal distillation performance if we naively enforce feature matching loss for each spatial location. We will investigate the use of a learned attention pooling module to learn to match the feature maps of the teacher and student model.
- A sufficiently large dataset might be needed to robustly train the student model. We will investigate using a subset of the full 100M histopathology images to train the student model as we need to strike a balance between reaching good performance and computational cost.
- There is no publicly available unified implementation of various distillation methods (to my knwoledge) that can be scaled to large datasets of 10 - 20M images with support for distributed training. We may need to implement our own distillation pipeline with the help of frameworks such as PyTorch Lightning or HuggingFace accelerate.

## Proposed timeline and experiments

We propose the following timeline for our project:

- 1st week: set-up distllation pipeline with support for ddp, organize subset of 100M images implement and train with global feature matching loss (matching just the global representation between student and teacher) and train student model on a subset of 10 - 20M images from the full set of 100M histopathology images.
- 2nd week: implement and train dense feature map matching with attention pooling module
- 3rd week: evaluate on 1 - 2 downstream supervised learning tasks, benchmark latency/throughput vs. accuracy trade-off
- 4th week: analysis and write-up
