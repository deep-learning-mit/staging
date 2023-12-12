---
layout: distill
title: A Deeper Look into Equivariance for Materials Data
description: A Comparative Analysis of an SE(3) Equivariant GNN and a Non-Equivariant GNN in Materials Data Tasks with a Focus on Investigating the Interpretability of Latent Geometry within the Two GNNs.
date: 2023-11-08
htmlwidgets: true

# Anonymize when submitting
# authors:
#   - name: Anonymous

authors:
  - name: Nofit Segal
    affiliations:
      name: MIT - CSE & DMSE


# must be the exact same name as your blogpost
bibliography: 2023-11-08-A-deeper-look-into-equivariance-for-materials-data.bib  

# Add a table of contents to your post.
#   - make sure that TOC names match the actual section names
#     for hyperlinks within the post to work correctly.
toc:
  - name: Introduction
  - name: Data
  - name: Comparative Analysis

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

Materials encompasses diverse chemical and physical properties, intricately influencing their suitability for various applications. Representing materials as graphs, with atoms as nodes and chemical bonds as edges, allows for a structured analysis. Graph Neural Networks (GNNs) emerge as promising tools for unraveling relationships and patterns within materials data. Leveraging GNNs can lead to the development of computational tools facilitating a deeper comprehension and design of structure-property relationships in atomic systems.

In the three-dimensional Euclidean space, materials, and physical systems in general, naturally exhibit rotation, translation, and inversion symmetries. When adopting a graph-based approach, a generic GNN may be sensitive to these operations, but an SE(3) equivariant GNN excels in handling such complexities. Its inherent capability to navigate through rotations, translations, and inversions allows for a more nuanced understanding, enabling the capture of underlying physical symmetries within the material structures.



## Data

Creating a dataset for this project will involve curating small molecules data samples, and generating diverse rotational and translational placements for analysis.  

<div class="row mt-3">
      {% include figure.html path="assets/img/2023-11-08-A-deeper-look-into-equivariance-for-materials-data/NH3_rot.png" class="img-fluid rounded z-depth-1" %}
      </div>
<div class="caption">
    Rotations of Ammonia (NH3) molecule 
</div>

Scalar properties, such as Energy, remain unaffected by the molecule's rotations. In contrast, directional properties like forces and moments undergo rotation along with the molecule's reorientation.


## Comparative Analysis

This project involves constructing two GNN architectures—one generic utilizing pytorch.geometric and the other SE(3) equivariant employing e3nn-torch—and comparing their performance in predicting molecular properties. The comparison will delvie into these critical aspects:


**Generalization**: Does either model demonstrate better generalization to unseen data?


**Interpretability**: Are there differences in the latent spaces geometry of the two models, and if so, how? This involves comparing the presence of clusters, their sizes, and their alignment with specific attributes.


**Data Efficiency**: How does each model's performance scale with datasets of varying sizes? Does one model exhibit superior predictive capabilities, particularly when faced with limited data?






