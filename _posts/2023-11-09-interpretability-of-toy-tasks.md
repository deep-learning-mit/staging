---
layout: distill
title: Studying Interpretability of Toy Models on Algirithmic Tasks 
description: This blog makes the case for the importance of studying small models on easy algorithmic tasks, in order to understand larger and more complicated networks.
date: 2023-11-08
htmlwidgets: true

# Anonymize when submitting
# authors:
#   - name: Anonymous

authors:
  - name: Vedang Lad 
    url: "https://www.vedanglad.come"
    affiliations:
      name: MIT


# must be the exact same name as your blogpost
bibliography: 2023-11-09-interpretability-of-toy-tasks.bib  

# Add a table of contents to your post.
#   - make sure that TOC names match the actual section names
#     for hyperlinks within the post to work correctly.
toc:
  - name: Research Question 
  - name: Outline of Work



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
### Research Question

Deep learning is seriously cool - the use of larger models, more data, and intricate architectures has led to the development of astonishingly powerful models capable of achieving the unimaginable. However, the added complexity raises a perplexing question: when we ask _how_ the model arrives at its solutions, we often find ourselves scratching our heads. This is where the concept of interpretability and explainability of models steps in.

There exists a body of work dedicated to investigating the interpretability of vision models. Researchers have delved into the intermediate layers of these models, uncovering the roles of different neurons and examining activations across various images.

To fathom modern **deep** learning, this project sets out to explore how these models actually learn. Specifically, it aims to understand how models uncover algorithms to tackle various simple tasks. The driving force behind this exploration is the belief that studying simple tasks in smaller, controlled settings can shed light on more extensive and intricate techniques. The project will employ straightforward architectures, such as lightly layered RNNs, compact MLPs, and single-layer transformers, for basic algorithmic tasks. These tasks may include, but are not confined to, bitwise addition, locating the minimum (or maximum) in a list, and rearranging lists. Essentially, the aim is to examine how we can utilize simplified models for simple algorithmic tasks to gain deeper insights into the workings of Large Language Models (LLMs) and complex architectures.

### Outline of Work

Depending on the time available, I may narrow the focus down to a single task and delve deeper into its exploration, for example, list permutation. The project will follow a progression in complexity, starting with results from a compact MLP, then transitioning to an RNN, and finally examining a simple transformer.

I intend to apply techniques covered in lectures, such as the analysis of Principal Component Analysis (PCA) on the internal activations of a transformer. Visualizing the activations of trained networks presents an exciting opportunity for captivating visual representations. One intriguing idea I have in mind is to demonstrate how the model's weights and activations evolve as the model learns.

Furthermore, I will draw from our class material by showcasing the use of intermediate embeddings within networks to illustrate how they discover algorithms to solve tasks.

In the end, the project will conclude by discussing the broader implications of this research. Although Large Language Models have displayed proficiency in simple mathematical calculations, this study will explore the point at which transformers face challenges in terms of complexity.

Prior research in the realm of model interpretability, such as the "The Clock and Pizza" paper ([https://arxiv.org/abs/2306.17844](https://arxiv.org/abs/2306.17844)) and the work on modular addition available here ([https://pair.withgoogle.com/explorables/grokking/](https://pair.withgoogle.com/explorables/grokking/)), will be referenced to provide context and build upon existing knowledge.

The overarching goal of this project is to reveal that neural networks don't have to remain mysterious black boxes. While machine learning has recently evolved into an engineering discipline, I aspire to illustrate through my project that unveiling the inner workings of these models can be approached as a scientific endeavor, much like neuroscience for computers.
