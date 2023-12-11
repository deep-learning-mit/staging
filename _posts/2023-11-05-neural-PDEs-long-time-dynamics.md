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

## Partial differential equations 
At the continuum level, spatiotemporal physical phenomena such as reaction-diffusion processes and wave propagations can be described by partial differential equations (PDEs). By modeling PDEs, we can understand the complex dynamics of and relationships between parameters across space and time, particularly at the mesoscale. However, PDEs usually do not have analytical solutions and are often solved numerically using methods such as the finite difference, finite volume, and finite element methods <d-cite key="LoggMardalEtAl2012"></d-cite>. For example, the finite element method (FEM) approximates PDE solutions by first discretizing a sample domain into a mesh of interconnected elements and then solving a system of equations iteratively given a set of boundary conditions, initial conditions, and material properties. 

The Navier-Stokes equation describes the dynamics of viscous fluids. For example, the PDE below is for viscous and incompressible fluid in vorticity form. The solution data were from the original paper<d-cite key="li2020fourier"></d-cite> where the equation is for a sampled initial condition, a constant forcing term, and periodic boundary conditions. The problem was solved with a pseudospectral method using a 1e-4 time step with the Crank-Nicolson scheme. 

$$
\begin{gather}
\partial_t w(x, t) + u(x, t) \cdot \nabla w(x, t) = \nu \Delta w(x, t) + f(x), \quad x \in (0,1)^2, t \in [0,T] \\
\nabla \cdot u(x, t) = 0, \quad x \in (0,1)^2, t \in [0,T] \\
w(x, 0) = w_0(x), \quad x \in (0,1)^2
\end{gather}
$$

You can visualize the solution for the Navier-Stokes equation through the time steps:

<div class="l-body-outset" style="display: flex; justify-content: center; align-items: center;">
  <iframe src="{{ 'assets/html/2023-11-05-neural-PDEs-long-time-dynamics/navierstokes.html' | relative_url }}" frameborder="0" scrolling="no" height="600px" width="100%"></iframe>
</div>
<div class="caption">
Solution of Navier-Stokes <d-cite key="li2020fourier"></d-cite>, drag the slider!
</div>

### Motivations for neural PDEs
Well-established numerical methods are very successful in approximating the solutions of PDEs, however, these methods require high computational cost especially for high spatial and temporal resolutions. Furthermore, it is important to have fast and accurate surrogate models that would target problems that require uncertainty quanitifcation, inverse design, and PDE-constrained optimizations. In recent years, there have been growing interests in neural operators that learn the mapping between input and output solution functions. These models are trained on numerical solutions from existing methods and inferences are orders of magnitude faster than calculating the solutions again through numerical methods. 

In my project, I will focus on these 3 questions:
1. **Learning dynamics in PDEs**: Can existing neural PDE methods effectively learn and generalize the underlying spatiotemporal dynamics of PDEs while ensuring consistent accuracies across various parameters and resolutions? 
2. **Representation learning in neural PDEs**: What unique representations do neural PDE methods learn in comparison to convolutional layers, and how do these new encodings influence PDE learning? 
3. **Improving long-term predictions**: How can neural PDE models be improved to predict long-term dynamics autoregressively while minimizing loss in accuracies?


## Base model: UNets with convolutional layers 
Let's begin with examining whether a U-Net with convolutional layers can be used to learn the dynamics. We can use Conv2d layers to learn features from each frame of the PDE solution, treating the solution in xy like an image. As for the time component, the surrogate model takes the input solution from the previous N time steps to predict solution in the next N+1 time step. Then, the solution from the previous N-1 steps are concatenated with the predicted next-step solution for input back into the model to predict the next step. In a nutshell, the model is trained to predict autoregressively. 

{% include figure.html path="assets/img/2023-11-05-neural-PDEs-long-time-dynamics/unet_train_test_loss.png" class="img-fluid" %}

{% include figure.html path="assets/img/2023-11-05-neural-PDEs-long-time-dynamics/unet_2dt_nspred42.gif" class="img-fluid" %}


## Applying FNO3d and FNO2dt on 2D Navier Stokes time-dependent PDE 
Fourier neural operators (FNOs) <d-cite key="li2020fourier"></d-cite> try to learn the mapping between input functions and solution functions <d-cite key="kovachki2021neural"></d-cite>, for example, mapping the solutions from earlier to later time steps for time-dependent PDEs. FNOs introduce the Fourier layers to learn convolution operators in the frequency domain by performing fast Fourier transforms and inverse transforms. 

Learning with FNO2d recurrent over time 

{% include figure.html path="assets/img/2023-11-05-neural-PDEs-long-time-dynamics/2dt_nspred42.gif" class="img-fluid" %}


{% include figure.html path="assets/img/2023-11-05-neural-PDEs-long-time-dynamics/train_test_loss.png" class="img-fluid" %}

{% include figure.html path="assets/img/2023-11-05-neural-PDEs-long-time-dynamics/nspred42.gif" class="img-fluid" %}

It learns the global dynamics well but these seem to be mostly global dynamic changes. 


### Representation learning: The learned representations in the Fourier layers 

{% include figure.html path="assets/img/2023-11-05-neural-PDEs-long-time-dynamics/fourierlayers.png" class="img-fluid" %}

{% include figure.html path="assets/img/2023-11-05-neural-PDEs-long-time-dynamics/convlayers.png" class="img-fluid" %}

How Fourier layers work in contrast with CNN layers, and why they can learn the underlying dynamics regardless of data resolution.

### Frequency modes that are learnt and the importance of positional encoding 

<div style="display: flex; justify-content: center; align-items: center;">
    {% include figure.html path="assets/img/2023-11-05-neural-PDEs-long-time-dynamics/show_dxdt.png" class="img-fluid" %}
    {% include figure.html path="assets/img/2023-11-05-neural-PDEs-long-time-dynamics/show_dydt.png" class="img-fluid" %}
</div>

Removing positional encodings for x and y grids would make the performance worse compared to with positional encodings. 

<div style="display: flex; justify-content: center; align-items: center;">
    {% include figure.html path="assets/img/2023-11-05-neural-PDEs-long-time-dynamics/noposencoding_dxdt.png" class="img-fluid" %}
    {% include figure.html path="assets/img/2023-11-05-neural-PDEs-long-time-dynamics/noposencoding_dydt.png" class="img-fluid" %}
</div>

{% include figure.html path="assets/img/2023-11-05-neural-PDEs-long-time-dynamics/train_test_loss_noencoding.png" class="img-fluid" %}

{% include figure.html path="assets/img/2023-11-05-neural-PDEs-long-time-dynamics/ns_noencode_pred42.gif" class="img-fluid" %}


## Inability to capture local dynamics and long-term accuracies in time-dependent PDEs
While the FNO has worked accurately for the Navier-Stokes data example, it does not perform well on other PDEs, especially when local dynamics and long-term accuracies are important. Here, I introduce another PDE - a coupled reaction heat-diffusion PDE with two dependent states. When the initial conditions of T and alpha are changed, the dynamics can become chaotic over time. 

$$
\begin{gather}
\kappa \frac{\partial^2 T}{\partial x^2} + \rho H_r \frac{\partial \alpha}{\partial t} = \rho C_p \frac{\partial T}{\partial t} \\
\frac{\partial \alpha}{\partial t} = A \exp \left( -\frac{E}{RT} \right) (1 - \alpha)^n 
\end{gather}
$$

<div class="l-body-outset">
  <iframe src="{{ 'assets/html/2023-11-05-neural-PDEs-long-time-dynamics/unstablefromp.html' | relative_url }}" frameborder='0' scrolling='no' height="750px" width="100%"></iframe>
</div>
<div class="caption">
Solution of the above coupled PDE with 2 dependent states, solved using FEM. Drag the slider!
</div>


Firstly, it can be harder for the Fourier layers to learn the local changes since the Fourier layers would only approximate kernels in the lower frequency modes and higher frequency modes are truncated away. Secondly, since numerical methods can be expensive, we want to use the first k steps (i.e. first 10 steps) of the true solution to predict the next N steps (as high as possible). Clearly, the prediction accuracies lower as we want higher resolution predictions for longer time steps as output. In an autoregressive training scheme, where the k input steps are used to predict the next step autoregressively until N steps are predicted on rollout, the losses will accumulate as we propagate more time steps forward. 

To overcome these 2 problems, there have been attempts to generally improve the accuracies of neural PDE models and also training tricks proposed to improve long-term accuracies in rollout. There were some techniques that were introduced in the paper on message passing neural PDEs <d-cite key="brandstetter2022message"></d-cite>, particularly the pushforward and the temporal bundling tricks. 




<div style="display: flex; justify-content: center; align-items: center;">
    {% include figure.html path="assets/img/2023-11-05-neural-PDEs-long-time-dynamics/516.gif" class="img-fluid" %}
    {% include figure.html path="assets/img/2023-11-05-neural-PDEs-long-time-dynamics/1129.gif" class="img-fluid" %}
    {% include figure.html path="assets/img/2023-11-05-neural-PDEs-long-time-dynamics/1526.gif" class="img-fluid" %}
</div>


I will first incorporate these techniques with the base FNO model to examine accuracies of long time dynamics. Next, I will examine if attention mechanisms introduced with transformer layers can help improve accuracies for lower frequency modes at longer time scales. These new model architectures would all be compared on the base dataset of Navier-Stokes equations (2D spatially with time dependence). 

### Using ReVIN to normalize and denormalize the time series input for 1D PDEs 


### Improving long time-step rollout with temporal bundling and pushforward tricks 


## Large Kernel Attention 




## Conclusion 
By the end of the project, I hope to propose a new neural PDE approach with **(1) spatial convolutions in the Fourier space, (2) transformer layers for improved attention on local dynamics, and (3) pushforward and temporal bundling tricks to deal with predictions over long time scales**. 