---
layout: distill
title: Exploring when convex optimization improves the generalization of deep neural networks
description: Recent work has shown how to embed convex optimization as a subroutine in the training of deep neural networks. Given that we can backpropagate through this procedure, the authors refer to this method as “convex optimization as a layer” leading to new neural network architectures. In machine learning, these deep networks can be used to solve a variety of problems: (1) in supervised learning, learn a classifier; (2) in reinforcement learning, learn a policy; (3) in generative modeling, learn a score function. We explore in each of these settings if a network architecture parameterized with convex optimization layers has an edge over off-the-shelf architectures like MLPs, CNNs, or U-Nets. The reader will take away a better understanding of when such an architecture could be useful to them given their data modality and prediction task. 


date: 2023-11-09
htmlwidgets: true


# Anonymize when submitting
# authors:
#   - name: Anonymous


authors:
 - name: Ram Goel
   affiliations:
     name: MIT CSAIL
 - name: Abhi Gupta
   affiliations:
     name: MIT CSAIL


# must be the exact same name as your blogpost
bibliography: 2023-11-09-convex_optimization_layer_deep_learning.bib 


# Add a table of contents to your post.
#   - make sure that TOC names match the actual section names
#     for hyperlinks within the post to work correctly.
toc:
 - name: Convex Optimization as a Layer in Neural Network Architectures
 - name: The Role of Convex Optimization Layers for Various Machine Learning Tasks
   subsections:
     - name: Supervised Learning
     - name: Reinforcement Learning
     - name: Generative Modeling


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


## Convex Optimization as a Layer in Neural Network Architectures


Convex optimization is a well-studied area of operations research. There has recently been an insurgence of work relating the field to machine learning. Agrawal et al. <d-cite key = "agrawal2019differentiable"></d-cite> propose a method known as ``disciplined parameterized programming’’, which maps the parameters of a given convex program to its solution in a differentiable manner. This allows us to view instances of convex optimization programs as functions mapping problem-specific data (i.e input) to an optimal solution (i.e output). For this reason, we can interpret a convex program as a differentiable layer with no trainable parameters in the same way as we think of ReLU as a layer in a deep neural network. Past work (<d-cite key = "amos2021optnet"></d-cite>, <d-cite key = "barratt2019differentiability"></d-cite>) has primarily focused on providing methods for differentiability of the convex optimization layer. However, an unexplored question remains: for which types of machine learning problems does this architecture provide an edge over other architectures?




## The Role of Convex Optimization Layers for Various Machine Learning Tasks


We hypothesize that architectures which leverage convex optimization layers may perform better on some machine learning tasks than others. CNNs have become the gold standard for solving supervised learning prediction tasks from image data. Transformers are now the go-to architecture in generative modeling when working with language. However, it remains unclear in which settings, if any, we may rely on convex optimization layers as the default choice of architecture. 


This project explores when such an architecture might be well-suited in machine learning. Specifically, we will implement a disciplined parametrized program for three separate tasks, in very different types of machine learning problems. We will then compare the performance of convex optimization as a layer between these tasks, using various metrics and baselines. This will provide insight as to which machine learning tasks are best suited for architectures with convex optimization layers. 


### Supervised Learning


We consider the supervised learning problem of predicting the solution to a sudoku puzzle from its image representation. We will compare against baseline CNN or MLP models, and we will compare the accuracy and amount of training needed across these architectures. We will render solutions to sudoku puzzles in  the context of convex optimization, and we hypothesize that the inductive bias of our architecture will provide better performance from existing architectures. In particular, we hypothesize that convex optimization as a layer will require less training and higher accuracy than for MLP and CNN architectures. 


### Reinforcement Learning


We consider the control problem of steering a car above a hill, otherwise known as MountainCar, from the OpenAI gym benchmark of RL environments. We can model the problem with quadratic reward, and linear transition function, so that the optimal controller would be quadratic in state. By contextualizing the action as a solution to a convex optimization problem, we can enforce safety constraints explicitly, for stability of training of the agent. We will compare this model against baseline RL algorithms such as PPO, and will compare standard RL metrics, such as mean reward. 


### Generative Modeling


We consider the generative learning problem of sampling maps for atari video games which satisfy specific conditions, such as the location of blocks or coins. We can make the data samples be solutions to an optimization problem, which enforces certain constraints on the generated solution, such as the locations or colors of features in the game. Then, by adding noise, and predicting the mean of noisy samples, we can generate fresh valid configurations also satisfying our optimization constraints. We will test the accuracy of our architecture by testing its accuracy across various tests and environments.  















