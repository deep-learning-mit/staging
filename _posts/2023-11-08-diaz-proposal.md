---
layout: distill
title: 'Vision Transformers: High-Frequency means High-Fidelity'
description: 'Vision Transforms have a quadratic complexity for the patch length. Past work have circumnavigated this complexity at the cost of losing information. Recent advances propose ViT amendments serving to preserve global attention and high-frequency information - all with a lowered computational burden. Here, we propose to investigate the translation of such architectures to a longstanding image restoration problem: MRI.'
date: 2023-11-08
htmlwidgets: true

authors:
  - name: Sebastian Diaz
    url:
    affiliations:
      name: MIT

# must be the exact same name as your blogpost
bibliography: 2023-11-08-diaz-proposal.bib  

toc:
  - name: Proposal Motivation
  - name: Proposal Outline
  - name: 'Vision Transformers: How, What, Why?'
    subsections:
      - name: Attention
      - name: Advantages over CNNs
      - name: Computational Complexity
  - name: Multi-Scale Windowed Attention
    subsections:
      - name: Swin Transformer
      - name: Other Approaches
  - name: Frontiers of ViT's
    subsections:
      - name: Global Context ViT's
      - name: Wavelet ViT's
  - name: ViT's in Image Restoration and MRI Reconstruction
    subsections:
      - name: SwinIR and SwinMR
      - name: New reconstruction architectures


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

## Proposal Motivation

Vision transformers (ViTs)<d-cite key="Dosovitskiy2020"></d-cite> have become increasingly popular in computer vision applications over the past years and have demonstrated state-of-the-art performance on many classification tasks compared to convolutional neural networks (CNNs). Due to the attention mechanism, ViT’s have the ability to learn more uniform short and long-range information. Despite this benefit, ViTs suffer from an increased computational complexity $\mathcal{O}(n^{2})$ with respect to the input patch number. This suffices for low-resolution images, but quickly becomes burdensome for high-resolution applications. There have been many advances seeking to balance between the computational complexity and the short and long-range spatial dependencies. One popular example is the Swin Transformer<d-cite key="Liu2021"></d-cite> that employs a convolutional-like approach by limiting self-attention to local windows and linearizing the complexity - at the cost of losing long-range information. The Swin Architecture can be seen in [Figure 1](#figure-1). Other approaches have used down-sampling techniques such as average or max pooling over the keys and values to reduce the complexity. However, these processes are non-invertible resulting in the loss of high-frequency components. In order to preserve the amount of information we extract from our image, recent work has incorporated a Wavelet block as a drop-in replacement for these down-sampling operations<d-cite key="Yao2022"></d-cite>. The Wavelet block consists of an invertible transform that breaks an image down into high and low frequency spatial components. Due to the inverse nature of such operations, high-frequency components of the image will be preserved. Another novel approach applies a hybrid attention scheme consisting of local and global self-attention modules. In each module, a global query token is generated and interacts with the local key and value tokens<d-cite key="Hatamizadeh2023"></d-cite>. 

These new approaches highlight the pursuit to preserve high-frequency features and long-range information - while simultaneously enabling increased performance. They entertain creative new ideas that warrant further investigation to completing characterize their potential in relavant domains. Therefore, I propose to investigate and analyze such architectures in MRI reconsturction where maintaining fidelity of the resultant image is essential to an individual's health. 

In more detail, I will investigate how each architecture can be coincided with the image restoration framework, SwinIR<d-cite key="Liang2021"></d-cite>. First, I will investigate the Wave-ViT, as it utilizes the long-studied Wavelet transform, which historically initiated the rapid acceleration of MRI images in the late 2000s when the field of compressed sensing met deliberate undersampling<d-cite key="Lustig2007"></d-cite>. The GC-ViT will also be studied in its ability to provide adequate MRI reconstruction while preserving detail. Both architectures will be compared to the most popular attention reconstruction network, SwinMR<d-cite key="Huang2022"></d-cite>. The data utilized will come from the field standard, fastMRI<d-cite key="Zbontar2018"></d-cite>, which was released by MetaAI and NYU.

<div class="col-sm">
    <a name="figure-1"></a>
    {% include figure.html path="assets/img/2023-11-08-diaz-proposal/swinvit.png" class="img-fluid rounded z-depth-1" %}
    <div class="caption">
        Figure 1: Swin Transformer Architecture.
    </div>
</div>

## Proposal Outline
1. Introduce Vision Transformers and their advantages and shortcomings when compared to CNNs
* The comparison will be visually illustrated. There will be a focus on the attention mechanism and its ability to adhere to multiple parts of the image.
2. Delve deeper into ViT's
* I will overview the current methods that are employed to reduce the computational complexity.
* There will be an emphasis on the Swin Transformer as it has historically served as a foundational for the rest of the hierarchical/multi-scale ViT approaches.
* A comparison between the ViT and Swin Transformer will be made.
3. Focus on the two recently proposed methods: GC-ViT's and Wave-ViT
* Plots and figures will be generated to demonstrate their potency and pitfalls.
* Diagrams will be generated for the reader to easily digest the creative approaches proposed by the authors
4. MRI Reconstruction will be introduced with current deep learning methods being overviewed.
* The SwinIR and SwinMR will be the focus, as they are a blueprint for further improvements and will give merit to the project's end goals in investigating each new approach. Their ability to solve an inverse problem will be a focal point.


**Additionally, see the Table of Contents for a preliminary structured outline.**

## Vision Transformers: How, What, Why?
### Attention
### Advantages over CNNs
### Computational Complexity
## Multi-Scale Windowed Attention
### Swin Transformer
### Other Approaches
## Frontiers of ViT's
### Global Context ViT's
### Wavelet ViT's
## ViT's in Image Restoration and MRI Reconstruction
### SwinIR and SwinMR
### New reconstruction architectures
