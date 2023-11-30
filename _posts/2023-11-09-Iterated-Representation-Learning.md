---
layout: distill
title: Iterated Representation Learning
description: Representation learning is a subfield of deep learning focused on learning meaningful lower-dimensional embeddings of input data, and rapidly emerging to popularity for its efficacy with generative models. However, most representation learning techniques, such as autoencoders and variational autoencoders, learn only one embedding from the input data, which is then used to either reconstruct the original data or generate new samples. This project seeks to study the utility of a proposed iterated representation learning framework, which repeatedly trains new latent space embeddings based on the data outputted from the last round of representation. In particular, we seek to examine whether the performance of this iterated approach on a model and input dataset are indicative of any robustness qualities of the model and latent embedding space, and potentially derive a new framework for evaluating representation stability.
date: 2023-11-09
htmlwidgets: true

# Anonymize when submitting
# authors:
#   - name: Anonymous

authors:
  - name: Angela Li
    url: "https://www.linkedin.com/in/angelayli/"
    affiliations:
      name: Harvard University
  - name: Evan Jiang
    url: "https://www.linkedin.com/in/evanjiang1/"
    affiliations:
      name: Harvard University

# must be the exact same name as your blogpost
bibliography: 2023-11-09-Regularization.bib  

# Add a table of contents to your post.
#   - make sure that TOC names match the actual section names
#     for hyperlinks within the post to work correctly.
toc:
  - name: Background
  - name: IRL Framework
    subsections:
    - name: IRL for AEs
    - name: IRL for VAEs
  - name: Potential Questions and Hypotheses
  - name: Future Work

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

## Project Proposal Overview

Welcome to our project proposal homepage! Below is an overview of what we're interested in and how we plan on structuring our project, as well as some questions included at the bottom that we hope to get some advice/feedback/input on.

### Background

1. Representation Primer
- What is representation?
- Why is it important to learn well (properties of good representations and its utility)?

2. Autoencoder Primer
- What is an autoencoder (AE) and how does it relate to representation?

### Iterated Representation Learning (IRL) Framework

1. AEs (deterministic reconstruction)
- Step 1: Given some dataset, use an AE to learn its embedding space.
- Step 2: Using the learned embedding and AE, reconstruct the original dataset and compute the reconstruction loss.
- Step 3: Using the reconstructed dataset, repeat Steps 1 and 2, iterating as long as desired.

2. VAEs (generative modeling)
- Step 1: Given some dataset, use a VAE to learn its embedding space.
- Step 2: Using the learned embedding and VAE, generate a new dataset.
- Step 3: Using the newly generated dataset, repeat Steps 1 and 2, iterating as long as desired.

### Potential Questions and Hypotheses
1. Following the iterated representation learning framework above, can we iterate until we reach some kind of convergence with respect to the model and/or learned embedding space? 
- If so, can this tell us any properties of the representation space, learned representation, model, and/or data? 
- Does the number of iterations until convergence have anything to do with how “good” or stable the model or learned representation is?
2. In the deterministic autoencoder case, how do the reconstruction losses perform as iterations go on? Do we converge? How quickly? If the loss seems to diverge (relative to the original data), does it diverge linearly, exponentially, etc.?
3. What can we say about characteristics of the data that are maintained through iterations, and characteristics that evolve as the iterations go on? 
- For example, if we observe that a model remains invariant to a certain feature, but becomes sensitive to new features of the data, what does this tell us about these particular features, our model, and the original data itself? 
- Are there any other patterns we can identify along these lines?
4. Can we propose some sort of representation learning evaluation framework using iterated representation learning, e.g. rough guidelines on ideal number of iterations required until convergence, and what this says about how good a model is? 

### Future Work
1. How can we make iterated representation learning more computationally tractable? 
2. Can any of these results be generalized to other types of deep learning models?
3. Are there any theoretical guarantees we can prove? 

## References and Resources

### Possible Data Sources

- MNIST, FashionMNIST
- CIFAR-10, CIFAR-100
- Pytorch’s Food101 dataset, CelebA dataset
- Tensorflow’s cats_vs_dogs dataset

### Possible References

- Robustness of Unsupervised Learning Without Labels (Petrov and Kwiatkowska, 2022) 
- Understanding Robust Learning through the Lens of Representation Similarities (Cianfarani et al., 2022)
- Using variational autoencoders to learn variations in data (Rudd and Wild, 2018)

## Questions for Course Staff

1. Does this problem seem tractable, both theoretically and empirically? 
2. Our idea encompasses two analogous processes, a deterministic pipeline with reconstruction (using an AE), and a random pipeline with new data generation (using a VAE). Do you think either of these is more/less practical, feasible, or interesting to pursue?
3. How would you recommend that we get started on this, beyond reading more existing literature on representation learning? We were thinking that perhaps we could try this approach on some smaller examples first (e.g. fixing a dataset and using a few different autoencoder models), and see if any interesting observations result from that, and then dive deeper based on those results. Any advice here would be greatly appreciated! 
4. Are there any theoretical components that you suggest we focus on, to potentially prove a small theoretical result?
5. What empirical results/comparisons would you suggest us to be on the lookout for?
6. Any other suggestions? 

