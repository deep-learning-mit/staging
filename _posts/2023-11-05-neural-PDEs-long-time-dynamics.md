---
layout: distill
title: Neural PDEs for learning local dynamics 
description: 6.898 deep learning project 
date: 2023-11-05
htmlwidgets: true

authors:
  - name: Pengfei Cai
    affiliations:
      name: MIT

# must be the exact same name as your blogpost
bibliography: 2023-11-05-neural-PDEs-long-time-dynamics.bib  

# Add a table of contents to your post.
#   - make sure that TOC names match the actual section names
#     for hyperlinks within the post to work correctly.
toc:
  - name: Partial differential equations and numerical methods 
  - name: Motivations for neural PDEs
  - name: Representation learning of dynamics with neural PDEs 
  - name: Improving predictions for long time-steps

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

## Partial differential equations and numerical methods 
At the continuum level, spatiotemporal physical phenomena such as reaction-diffusion processes and wave propagations can be described by partial differential equations (PDEs). By modeling PDEs, we can understand the complex dynamics of and relationships between parameters across space and time, particularly at the mesoscale. However, PDEs usually do not have analytical solutions and are often solved numerically using methods such as the finite difference, finite volume, and finite element methods <d-cite key="LoggMardalEtAl2012"></d-cite>. For example, the finite element method (FEM) approximates PDE solutions by first discretizing a sample domain into a mesh of interconnected elements and then solving a system of equations iteratively given a set of boundary conditions, initial conditions, and material properties. **In my project, I will first briefly introduce established methods of numerically solving PDEs.** A popular PDE is the Navier-Stokes equation that describes the dynamics of viscous fluids, an example PDE below is the 2D Navier-Stokes equation for viscous and incompressible fluid in vorticity form and I will use its numerical solutions as training data for my project. 

$$
\begin{gather}
\partial_t w(x, t) + u(x, t) \cdot \nabla w(x, t) = \nu \Delta w(x, t) + f(x), \quad x \in (0,1)^2, t \in [0,T] \\
\nabla \cdot u(x, t) = 0, \quad x \in (0,1)^2, t \in [0,T] \\
w(x, 0) = w_0(x), \quad x \in (0,1)^2
\end{gather}
$$

## Motivations for neural PDEs
Well-established numerical methods are very successful in approximating the solutions of PDEs, however, these methods require high computational cost especially for high spatial and temporal resolutions. Furthermore, it is important to have fast and accurate surrogate models that would target problems that require uncertainty quanitifcation, inverse design, and PDE-constrained optimizations. In recent years, there have been growing interests in neural operators that learn the mapping between input and output solution functions. These models are trained on numerical solutions from existing methods and inferences are orders of magnitude faster than calculating the solutions again through numerical methods. 

In my project, I will focus on these 3 questions:
1. **Learning dynamics in PDEs**: Can existing neural PDE methods effectively learn and generalize the underlying spatiotemporal dynamics of PDEs while ensuring consistent accuracies across various parameters and resolutions? 
2. **Representation learning in neural PDEs**: What unique representations do neural PDE methods learn in comparison to convolutional layers, and how do these new encodings influence PDE learning? 
3. **Improving long-term predictions**: How can neural PDE models be improved to predict long-term dynamics autoregressively while minimizing loss in accuracies?


## Base model: UNets with convolutional layers 

{% include figure.html path="assets/img/2023-11-05-neural-PDEs-long-time-dynamics/unet_train_test_loss.png" class="img-fluid" %}

{% include figure.html path="assets/img/2023-11-05-neural-PDEs-long-time-dynamics/unet_2dt_nspred42.gif" class="img-fluid" %}


## Applying FNO3d and FNO2dt on 2D Navier Stokes time-dependent PDE 
In my project, I will first examine the neural operator approaches in learning PDEs, particularly Fourier neural operators. Fourier neural operators (FNOs) <d-cite key="li2020fourier"></d-cite> learn the mapping between input functions and solution functions <d-cite key="kovachki2021neural"></d-cite>, for example, mapping the solutions from earlier to later time steps for time-dependent PDEs. FNOs introduce the Fourier layers to learn convolution operators in the frequency domain by performing fast Fourier transforms and inverse transforms. The animation below shows the predicted solutions for the Navier-Stokes equations using a pretrained FNO. 


{% include figure.html path="assets/img/2023-11-05-neural-PDEs-long-time-dynamics/train_test_loss.png" class="img-fluid" %}

{% include figure.html path="assets/img/2023-11-05-neural-PDEs-long-time-dynamics/nspred42.gif" class="img-fluid" %}

It learns the global dynamics well but these seem to be mostly global dynamic changes. 


{% include figure.html path="assets/img/2023-11-05-neural-PDEs-long-time-dynamics/2dt_nspred42.gif" class="img-fluid" %}

Learning with FNO2d recurrent over time 


### Representation learning: The learned representations in the Fourier layers 

{% include figure.html path="assets/img/2023-11-05-neural-PDEs-long-time-dynamics/fourierlayers.png" class="img-fluid" %}

{% include figure.html path="assets/img/2023-11-05-neural-PDEs-long-time-dynamics/convlayers.png" class="img-fluid" %}

How Fourier layers work in contrast with CNN layers, and why they can learn the underlying dynamics regardless of data resolution.

### Lower frequency modes are learned - sinusoidal waves 

{% include figure.html path="assets/img/2023-11-05-neural-PDEs-long-time-dynamics/show_dxdt.png" class="img-fluid" %}

{% include figure.html path="assets/img/2023-11-05-neural-PDEs-long-time-dynamics/show_dydt.png" class="img-fluid" %}


### Importance of positional encoding 
Removing positional encodings for x and y grids would make the performance worse compared to with positional encodings. 

{% include figure.html path="assets/img/2023-11-05-neural-PDEs-long-time-dynamics/train_test_loss_noencoding.png" class="img-fluid" %}

{% include figure.html path="assets/img/2023-11-05-neural-PDEs-long-time-dynamics/ns_noencode_pred42.gif" class="img-fluid" %}

{% include figure.html path="assets/img/2023-11-05-neural-PDEs-long-time-dynamics/noposencoding_dxdt.png" class="img-fluid" %}

{% include figure.html path="assets/img/2023-11-05-neural-PDEs-long-time-dynamics/noposencoding_dydt.png" class="img-fluid" %}

## Learning long term dynamics in coupled time-dependent PDE

However, I will then highlight the inability for FNOs to capture long term dynamics (the losses accumulate when predictions made autoregressively are needed for long time steps) and how it could be due to FNOs' inability to learn representations at lower frequency modes (especially for systems where we care more about local changes instead of global changes in the dynamics). These are still preliminary hypotheses which I will examine properly during the project. 


To deal with long time-steps predictions for time-dependent PDEs is still an open research question. There were some techniques that were introduced in the paper on message passing neural PDEs <d-cite key="brandstetter2022message"></d-cite>, particularly the pushforward and the temporal bundling tricks. I will first incorporate these techniques with the base FNO model to examine accuracies of long time dynamics. Next, I will examine if attention mechanisms introduced with transformer layers can help improve accuracies for lower frequency modes at longer time scales. These new model architectures would all be compared on the base dataset of Navier-Stokes equations (2D spatially with time dependence). Finally, I also hope to compare these methods on a coupled reaction heat-diffusion PDE with two dependent states, shown below. This PDE tend to have more chaotic behaviors when initial conditions are changed and thus its dynamics can be harder to learn especially for longer time scales and for lower frequency modes (for FNOs). 

$$
\begin{gather}
\kappa \frac{\partial^2 T}{\partial x^2} + \rho H_r \frac{\partial \alpha}{\partial t} = \rho C_p \frac{\partial T}{\partial t} \\
\frac{\partial \alpha}{\partial t} = A \exp \left( -\frac{E}{RT} \right) (1 - \alpha)^n 
\end{gather}
$$

Therefore, by the end of the project, I hope to propose a new neural PDE approach with **(1) spatial convolutions in the Fourier space, (2) transformer layers for improved attention on local dynamics, and (3) pushforward and temporal bundling tricks to deal with predictions over long time scales**. 