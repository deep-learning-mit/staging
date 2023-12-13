---
layout: distill
title: Investigating the Impact of Symmetric Optimization Algorithms on Learnability
description: Recent theoretical papers in machine learning have raised concerns about the impact of symmetric optimization algorithms on learnability, citing hardness results from theoretical computer science. This project aims to empirically investigate and validate these theoretical claims by designing and conducting experiments at scale. Understanding the role of optimization algorithms in the learning process is crucial for advancing the field of machine learning.
date: 2023-11-09
htmlwidgets: true

# Anonymize when submitting
# authors:
#   - name: Anonymous

authors:
  - name: Kartikesh Mishra
    url: ""
    affiliations:
      name: MIT
  - name: Divya P Shyamal
    url: ""
    affiliations:
      name: MIT

# must be the exact same name as your blogpost
bibliography: 2023-11-01-Symmetry-Optimization.bib  

# Add a table of contents to your post.
#   - make sure that TOC names match the actual section names
#     for hyperlinks within the post to work correctly.
toc:
  - name: Introduction
  - name: Experimental design
    subsections:
    - name: Learning Tasks and Datasets
    - name: Learning Algorithms
  - name: Evaluation Metrics

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

## Introductions

In practice, the majority of machine learning algorithms exhibit symmetry. Our objective is to explore the impact of introducing asymmetry to different components of a machine learning algorithm, such as architecture, loss function, or optimization, and assess whether this asymmetry enhances overall performance. 

Andrew Ng's research <d-cite key="ng2004feature"></d-cite> (https://icml.cc/Conferences/2004/proceedings/papers/354.pdf) suggests that in scenarios requiring feature selection, employing asymmetric (or more precisely, non-rotationally invariant) algorithms can result in lower sample complexity. For instance, in the context of regularized logistic regression, the sample complexity with the L1 norm is O(log n), while with the L2 norm, it is O(n). This insight underscores the potential benefits of incorporating asymmetry, particularly in tasks involving feature selection, to achieve improved learning outcomes. Can asymmetry be more advantageous in other learning tasks? What are the costs associated with using symmetric or asymmetric learning algorithms?

## Experimental Design

Our experiments will proceed as follows. We will have a set of datasets and a set of learning algorithms (both symmetric and asymmetric) from which we will generate models and test them on validation datasets from the same distribution on which they were trained. We will analyze the learning process as well as the performance of these learned models.

### Learning Tasks and Datasets

We plan to use MNIST, CIFAR-100, IRIS Datasets like Banknote Dataset, and a subset of ImageNet. If we complete our training on the image datasets, we may include some text-based datasets from Kaggle. Using these datasets, we plan to analyze several learning tasks: classification, regression, feature selection, and reconstruction. 

### Learning Algorithms

We define a gradient descent parametric learning algorithm to be symmetric if it uses the same function to update each parameter value. Currently, we are considering using CNN models with varying numbers of convolution layers, VisTransformers with varying numbers of attention blocks, and MultiLayer Perceptron with varying depths of the network. We will use dropout, skip connections, variation in activation functions, and initialization across layers to introduce asymmetry in the architecture. We will use cross-entropy and MSE Loss functions as asymmetric and symmetric loss functions. For our optimizers, we will use Batch Gradient Descent, Stochastic Gradient Descent, and Adam algorithms, and to introduce asymmetry, we will vary the learning rates, momentum, and weight decay across parameters. 

For our initial tests, we plan to compare a few pairs of multi-layer perceptions on the MNIST dataset. Each pair is described in detail below.

- 3-layer perceptron with l as learning rate  vs 3-layer perceptron with each layer k having lk learning rates
- 4-layer perceptron  vs 4-layer perceptron where some neurons on the 2nd layer skip to the 4th layer directly


## Evaluation Metrics

We will evaluate the trained models using the following metrics and compare the models generated from symmetric algorithms with those from asymmetric algorithms on the same dataset.
  - validation accuracy
    - Percentage of correct classifications
    - negative mean square error for regression and reconstruction
  - k-fold cross validation accuracy
  - accuracy on perturbed dataset (we will use guassian noise)
  - convergence speed during training

## Compute Resources

We plan to use Google Collab for our initial experiments and then use MIT Supercloud for training and
inference on large models.