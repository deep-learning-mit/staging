---
layout: distill
title: 6.S898 Project Proposal
description: A proposal for a research project that aims to develop a methodology to improve the robustness of contemporary neural networks. 
date: 2023-11-09
htmlwidgets: true

# Anonymize when submitting
# authors:
#   - name: Anonymous

authors:
  - name: Jackie Lin
    affiliations:
      name: MIT
  - name: Nten Nyiam
    affiliations:
      name: MIT

# must be the exact same name as your blogpost
bibliography: 2023-11-09-generating-robust-networks.bib

# Add a table of contents to your post.
#   - make sure that TOC names match the actual section names
#     for hyperlinks within the post to work correctly.
toc:
  - name: Introduction
  - name: Past Works
  - name: Data
  - name: Methodology
  - name: Timeline

# Below is an example of injecting additional post-specific styles.
# This is used in the 'Layouts' section of this post.
# If you use this post as a template, delete this _styles block.
---

## Introduction 
While deep neural networks can have state-of-the-art performance on image classification tasks, they are often vulnerable to adversarial perturbations. Their brittleness poses a significant challenge toward deploying them in empirical applications where reliability is paramount, such as medical diagnosis and autonomous vehicles. This project aims to assess the robustness of state-of-the-art neural networks for image classification by studying their vulnerability to adversarial perturbations and, subsequently, enhance their resilience through a combination of data augmentation and strategic fine-tuning.

## Past Works
To improve the resilience of contemporary neural networks, a foundational step involves comprehending how they work. Prior research diving into the visualization of neural network features <d-cite key="zeiler2013"></d-cite><d-cite key="simonyan2014"></d-cite><d-cite key="olah2017"></d-cite> will be particularly relevant for this step. To understand the weaknesses/brittleness of these neural networks, it would also be useful to reference works that study the generation of adversarial perturbations for images <d-cite key="szegedy2014"></d-cite><d-cite key="carlini2017"></d-cite>. 

## Data 
We will be using various neural networks pretrained on the ImageNet dataset, such as ResNet, VGGNet, and AlexNet. ImageNet is a dataset consisting over 14 million images and organized into over 20000 subcategories. Each image in the dataset is accompanied by detailed annotations, providing ground-truth data and allowing us to discern the objects and concepts featured in the images. ResNet, short for Residual Network, is a neural network that is best known for residual blocks, which enable training extremely deep networks while mitigating the vanishing gradient problem. Models like ResNet-50, ResNet-101, and ResNet-152 are renowned for their deep architectures and are widely used in various computer vision tasks. VGGNet, developed by the Visual Geometry Group (VGG), is known for its straightforward architecture. Models like VGG16 and VGG19 are characterized by a stack of convolutional layers and are widely used for various image analysis tasks. AlexNet is made up of five convolutional layers and three fully connected layers and played a significant role in popularizing deep learning for image classification.

## Methodology
First, we plan on developing a deep understanding of how each of the pretrained neural networks functions. In particular, we will use various visualization techniques to assess what features each network is learning in each layer. Then, we will assess the robustness of each network. Specifically, we will use perturbations like adding random Gaussian noise and greedily modifying pixels that impact classification the most to generate adversarial examples. Finally, the bulk of the project will be centered around leveraging the insights gained in the previous two steps to develop a data augmentation + fine-tuning procedure to make each of the neural networks more robust. One potential strategy involves freezing less brittle layers of the network and updating the weights of the more brittle layers by using adversarial examples as additional training examples. The ultimate goal is to devise a methodology that can be used to consistently generate more robust networks from existing networks. 

## Timeline 
- Nov 9, 2023: Submit the project proposal
- Nov 15, 2023: Read each of the related works carefully 
- Nov 20, 2023: Apply visualization techniques to each of the networks 
- Nov 24, 2023: Develop a procedure to generate adversarial examples for each network
- Dec 4, 2023: Visualize how the networks process adversarial examples, brainstorm and try out various strategies to improve robustness of network using insights gained 
- Dec 8, 2023: Consolidate and interpret results
- Dec 12, 2023: Submit the final project
