---
layout: distill
title:  A Transformer-Based Approach for Simulating Ecological Recovery

description: This project employs Transformers for a comprehensive spatial-temporal analysis of post-Mountaintop Removal landscape recovery, utilizing satellite imagery and DEMs. It focuses on integrating geomorphological changes to predict ecological succession. Advanced Transformer architectures will be used to enhance the interpretability of complex spatial features over time, aiming to create an accurate 3D simulation environment for interactive exploration and effective restoration planning.
date: 2023-12-12
htmlwidgets: true

# Anonymize when submitting
# authors:
#   - name: Anonymous

authors:
  - name: Crystal Griggs
    url: "https://crystalgriggs.com"
    affiliations:
      name: Massachusetts Institute of Technology

# must be the exact same name as your blogpost
bibliography: 2023-11-10-A-Transformer-Based-Approach-for-Simulating-Ecological-Recovery.bib

# Add a table of contents to your post.
#   - make sure that TOC names match the actual section names
#     for hyperlinks within the post to work correctly.
toc:
  - name: Introduction
    subsections:
    - name: Objective
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

## Introduction
This project focuses on the application of Transformer models to conduct a spatial-temporal analysis of terrain and vegetation informed by satellite imagery and Digital Elevation Models (DEMs), with an added focus on geomorphological phenomena such as erosion and terrain incisions. The utilization of Transformer architecture aims to supersede the capabilities of traditional models by exploiting the ability to understand the complex relationship between spatial features and temporal evolution. The work will exploit the temporal resolution and spectral diversity of the datasets to not only reconstruct the ecological succession post-Mountaintop Removal (MTR) in Appalachia, but also to simulate the geomorphological processes that shape the terrain over time. By integrating dynamic elements, the project looks to provide predictive insights for environmental monitoring and landscape restoration, ensuring a deeper understanding of both the ecological and geomorphological features of landscape recovery.

### Objective

Employing Transformer models, a detailed analysis of Digital Elevation Models and satellite imagery is used to simulate the ecological recovery of terrains impacted by Mountaintop Removal. It utilizes the Transformer's detailed analytical abilities, nown for its self-attention mechanisms, for precise land cover classification and to capture geomorphological changes, such as erosion and terrain incisions. These models excel in identifying patterns over time, critical for tracking the progression of natural regrowth and the effects of erosion. The combination of diverse datasets through the Transformer framework aims to generate an intricate and evolving 3D representation of the landscape, offering a clear depiction of its current state and potential recovery pathways, serving as an instrumental resource for informed environmental restoration and planning.

### Methodology

<u>Data Acquisition and Preprocessing</u>

The first stage will involve the collection of multi-spectral satellite imagery and high-resolution Digital Elevation Models (DEMs) of MTR-affected landscapes. This data will be preprocessed to ensure compatibility, which includes image normalization, augmentation, and the alignment of satellite imagery with corresponding DEMs to maintain spatial congruence. Preprocessing will also involve the segmentation of satellite data into labeled datasets for supervised learning, with categories representing different land cover types relevant to ecological states.

<u>Transformer Models for Spatial-Temporal Analysis</u>

Transformer models have exhibited remarkable success beyond their initial domain of natural language processing. Their unique self-attention mechanism enables them to capture long-range dependencies, making them a potentially good choice for complex spatial analysis. Vision Transformers, in particular, offer a new approach by treating image patches as tokens and allowing them to process the global context of an image effectively. This capability is beneficial for satellite imagery analysis, where understanding the broader environmental context is critical. Transformers designed for point cloud data, adapting to the inherent irregularities of LiDAR measurements, can potentially uncover intricate structural patterns and temporal changes within landscape data. With strategic approaches like transfer learning, transformers can overcome their computational resource complexity. 

<u>Visualization and Simulation</u>

The final step will be the development of a 3D simulation environment using Unreal Engine. The simulation will visualize the predicted ecological states and changes over time, providing an interactive tool for users to explore the landscape recovery process. The interface will allow users to manipulate variables and observe potential outcomes of different restoration strategies in a virtual setting.

### Evaluation

For the spatial analysis of satellite imagery and LiDAR data, the evaluation will focus on the transformer’s ability to discern and classify diverse land cover types. The key metrics for this assessment will include accuracy, precision, recall, and the F1 score extracted from confusion matrices. The model should accurately identify and categorize ecological features from high-resolution imagery. 
Temporally, the performance will be evaluated based on its capacity to predict ecological changes over time. This involves analyzing the model’s output against a time series of known data points to calculate the Mean Squared Error (MSE) for continuous predictions or log-loss for discrete outcomes. 

