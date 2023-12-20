---
layout: distill
title: Investigating Neural Operator Models for Closure Modeling of Fluid Dynamical Systems
description: Project Proposal for 6.s898 Deep Learning (Fall 2023)
date: 2023-11-08
htmlwidgets: true

# Anonymize when submitting
# authors:
#   - name: Anonymous

authors:
  - name: Anantha Narayanan Suresh Babu 
    url: "http://mseas.mit.edu/?p=5800"
    affiliations:
      name: MIT
  - name: Ruizhe Huang
    url: "https://ruizhe.tech/"
    affiliations:
      name: MIT

# must be the exact same name as your blogpost
bibliography: 2023-11-08-Investigating-neural-operator-models-for-closure-modeling-of-dynamical-systems.bib 

# Add a table of contents to your post.
#   - make sure that TOC names match the actual section names
#     for hyperlinks within the post to work correctly.
toc:
  - name: Background
  - name: Project Plan
  - name: Key Analyses and Investigations

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

# Background

Over the past decade, deep learning models have increasingly been used for modeling time series data for fluid dynamical systems. One of the most recent applications is in forecasting weather <d-cite key="schultz2021can"></d-cite> with deep learning models being developed by tech giants including Nvidia <d-cite key="pathak2022fourcastnet"></d-cite> and Google <d-cite key="lam2022graphcast"></d-cite> with reasonable prediction accuracy compared to conventional numerical weather prediction. While these models completely replace traditional numerical weather models with deep neural networks (i.e, \"surrogate modeling\"), in general, deep neural models can also be used to augment existing numerical solvers and methods <d-cite key="lino2023current"></d-cite>.

Training deep neural models to completely replace numerical solvers requires a lot of data, which might not be available due to constraints with sensor and satellite usage associated with collecting ocean and weather data. Additionally, these surrogate models are completely data-driven and could lead to non-physical predictions (lack of volume preservation, and non-conservation of physical laws) if these needs are not explicitly attended to during training <d-cite key="lino2023current"></d-cite>. A huge advantage of these models is their very low computational cost during inference compared to using numerical solvers <d-cite key="pathak2022fourcastnet"></d-cite>. Another approach is to use closure models that augment low fidelity (low resolution) numerical simulations with a neural network (i.e, a closure term) to predict high fidelity (high resolution) forecasts <d-cite key="gupta2021neural"></d-cite>. This approach could lead to some conservation of physical laws since it builds upon conventional numerical solvers that obey physical equations like PDEs, with a lower computational cost compared to directly running high fidelity numerical simulations.

# Project Plan

In this project, we plan to investigate the use of deep neural models like neural operators for closure modeling of dynamical systems. In particular, we plan to predict high resolution forecasts by augmenting low resolution numerical simulations with deep neural networks like neural operators.

We seek to find the deep neural network, $f_{NN}$, that best solves the equation
$$
u_{\text{high-res}}(\cdot)=u_{\text{low-res}}(\cdot) + f_{NN}(u_{\text{low-res}}(\cdot))
$$
where $u$ is the field of interest, 'high-res' and 'low-res' indicate high and low resolution numerical simulations and the $(\cdot)$ represents spatio-temporal coordinates. For $f_{NN}$, we plan to investigate the use of Fourier Neural Operators <d-cite key="li2020fourier"></d-cite>. These operators build upon Fourier kernels and directly learn the mapping between two infinite-dimensional function spaces, and have been used in various fluid dynamics applications as surrogate models. They key difference is that here we plan to use Fourier Neural Operators for closure modeling and not surrogate modeling, i.e., we will use the neural network to augment and not completely replace existing numerical PDE solvers.

{% include figure.html path="assets/img/2023-11-08-Investigating-neural-operator-models-for-closure-modeling-of-dynamical-systems/cloure_model.jpg" class="Img-closure-model" %}


We plan to use training and test data from numerical simulations of classical fluid flows like periodic eddy shedding from flow past a cylinder <d-cite key="cohen2004fluid"></d-cite>. If time permits, we would test our methodology on real surface velocity fields from ocean reanalysis data in the Massachusetts Bay, Mid-Atlantic Bight or the Gulf of Mexico. Hence, our training and test data would be 2D velocity fields at two resolutions (high and low) at various time instants. Both the velocity fields would be generated with identical initial and boundary conditions. The model accuracy would be judged by how close the prediction is compared to the high resolution ground truth (one choice is to use the RMSE or $L_2$ norm as the loss function, but there are other factors to consider, see below).

# Key Analyses and Investigations

The key analyses/ investigations we plan to do are:

1.  **Architectural choices and hyperparameters**: We will investigate the different choices of architecture, i.e., combination of Fourier Neural Operators with CNNs or vision transformers <d-cite key="pathak2022fourcastnet"></d-cite>. Our initial hypothesis is that CNNs might be better suited for this task since transformers are data hungry, and we have limited training data. We will also investigate the different positional embedding choices and usage of self vs
    cross-attention.
    
2.  **Training methodology and loss functions for long roll out**: We will investigate how to define loss functions (choice of error norms) and training approaches (using one time-step error as loss vs multi time-step error as loss) that would achieve low temporal roll out error since we deal with spatio-temporal dynamical systems, in which the prediction errors would accumulate during recursive forecasts for long time horizons <d-cite key="lippe2023pde"></d-cite>.
    
3.  **Pre-training latent representations:** If time permits, we will also investigate the usage of pre-training to learn good latent representations that help with closure modeling and accurate long roll out predictions. Here, we would compare the performance of multilayer perceptron autencoders, convolutional autoencoders (CAEs) or variational autoencoders (VAEs). However, care must be taken since all latent representations that lead to good decoder reconstruction accuracy, need not be well suited to the primary task of closure modeling <d-cite key="kontolati2023learning"></d-cite>.
    
4.  **Comparison with other closure modeling techniques:** Finally, depending on time constraints, we would like to compare the closure model obtained from using deep neural operators with those obtained by using other approaches like neural ODEs/ DDEs <d-cite key="gupta2021neural"></d-cite> or Gaussian Processes (a classical approach) for closure modeling <d-cite key="anh2000technique"></d-cite>.
