---
layout: distill
title: Project Proposal
# description: Your blog post's abstract.
#   This is an example of a distill-style blog post and the main elements it supports.
date: 2023-11-08
htmlwidgets: true

# Anonymize when submitting
# authors:
#   - name: Anonymous

authors:
  - name: Cathy Cai
    affiliations:
      name: MIT

# must be the exact same name as your blogpost
bibliography: 2023-11-08-overparameterization.bib  

# Add a table of contents to your post.
#   - make sure that TOC names match the actual section names
#     for hyperlinks within the post to work correctly.
toc:
  - name: Project Proposal
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

## Project Proposal

In my final project, I want to analyze the role of over-parameterization in the generalization of neural networks. Empirical work has demonstrated that over-parameterized neural networks generalize better to test data, which is counterintuitive because conventional wisdom states that overparameterized network can easily fit random labels to the data. Previous work has sought to explain this phenomena in MLPs and CNNs. The work of @neyshabur2018towards analyzed the capacity bound of two layer ReLU networks and demonstrates that it decreases with width. The work of @nichani2020increasing analyzed the test risk as depth increases with CNNs and showed that it follows a U-shaped curve. In my proposal, I want to analyze why another form of overparameterized neural networks do well: the Neural Tangent Kernel @cho2009kernel. The NTK approximates an MLP with infinite width and outperforms neural networks on certain tasks, e.g. @radhakrishnan2022simple. I want to analyze NTKs to assess whether the kernel-structure gives some information to the generalization capabilities of the extremely overparameterized neural networks. The key questions I want to answer include: why do overparameterized neural networks work so well? Is the wider the better? How does generalization capacity differ between types of models (e.g. NN/CNNs, NTK/CNTK)? 

### Outline
* Literature Review
* Looking at test risk and model capacity of kernel regression with different kernels (e.g. NTK + RELU, Laplacian, Gaussian) or Gaussian processes
* Some experiments demonstrating the role of overparameterization across different datasets across different methods