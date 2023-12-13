---
layout: distill
title: "Autodecoders: Analyzing the Necessity of Explicit Encoders in Generative Modeling"
description: The traditional autoencoder architecture consists of an encoder and a decoder, the former of which compresses the input into a low-dimensional latent code representation, while the latter aims to reconstruct the original input from the latent code. However, the autodecoder architecture skips the encoding step altogether and trains randomly initialized latent codes per sample along with the decoder weights instead. We aim to test the two architectures on practical generative tasks as well as dive into the theory of autodecoders and why they work along with their benefits.
date: 2023-11-09
htmlwidgets: true

# Anonymize when submitting
# authors:
#   - name: Anonymous

authors:
  - name: Seok Kim
    affiliations:
      name: Massachusetts Institute of Technology
  - name: Alexis Huang
    affiliations:
      name: Massachusetts Institute of Technology

# must be the exact same name as your blogpost
bibliography: 2023-11-09-autodecoders.bib

# Add a table of contents to your post.
#   - make sure that TOC names match the actual section names
#     for hyperlinks within the post to work correctly.
toc:
  - name: Outline
  - name: Background
  - name: Applications
  - name: Plan
  - name: References

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

### Outline

For our project, we plan to investigate the autodecoder network for generative modeling and its benefits and drawbacks when compared to the traditional autoencoder network. We will also explore the potential applications of autodecoders in various domains, particularly in 3D scene reconstructions.

### Background

Autoencoders have been extensively used in representation learning, comprising of the encoder network, which takes a data sample input and translates it to a lower-dimensional latent representation, and the decoder network, which reconstructs the original data from this encoding. By learning a compressed, distributed representation of the data, autoencoders greatly assist with dimensionality reduction.

In contrast, the autodecoder network operates without an encoder network for learning latent codes. Rather than using the encoder to transform the input into a low-dimensional latent code, each sample in the training set starts with a randomly initialized latent code, and the latent codes and the decoder weights are both updated during the training time. For inference, the latent vector for a given sample is determined through an additional optimization loop.

{% include figure.html path="/assets/img/2023-11-09-autodecoders/autoencoder_schematic.png" class="img-fluid" %}

_Image taken from "DeepSDF: Learning Continuous Signed Distance Functions for Shape Representation" by Park et al._

### Applications

One notable application of autodecoder networks is in 3D scene reconstructions. Traditional autoencoders tend to learn a single global latent code, making them less suitable for scenes with multiple objects and complex compositional structures. On the other hand, autodecoders can learn local latent codes, allowing for more efficient performance on scenes with multiple objects. This is particularly valuable in inverse graphics tasks to understand and reconstruct novel views of complex scenes.

### Plan

We will start by providing a detailed overview of how autodecoders function in a comprehensive blog post. This will include a thorough explanation of their architecture, training process, and potential applications. We will also discuss the theoretical advantages and disadvantages of autodecoder networks compared to traditional autoencoders.

Then, for the experimental part of our project, we will construct simple versions of both an autoencoder and an autodecoder network. These networks will be similarly trained and evaluated on a common dataset, such as the widely-used MNIST dataset, where we will attempt to generate novel images with both models. We will then conduct a comparative analysis of the performance of the two different networks, highlighting the differences in their performances and their respective strengths and weaknesses. This experiment will give us a good idea of the efficacy of the two different networks as well as how they compare to each other.

Additionally, we plan to assess whether one network performs better on out-of-distribution generalization tasks. By understanding the potential benefits and drawbacks of autodecoder networks, we can better leverage this innovative approach for a variety of generative tasks and gain insight into their applicability in a broader context.

### References

https://www.inovex.de/de/blog/introduction-to-neural-fields/

https://arxiv.org/pdf/1901.05103.pdf

https://karan3-zoh.medium.com/paper-summary-deepsdf-learning-continuous-signed-distance-functions-for-shape-representation-147af4740485
