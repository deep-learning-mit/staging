---
layout: distill
title: Neural PDEs for learning local dynamics and longer temporal rollouts
description: 6.s898 deep learning project 
date: 2023-11-05
htmlwidgets: true

authors:
  - name: Pengfei Cai
    affiliations:
      name: MIT

# must be the exact same name as your blogpost
bibliography: 2023-11-05-neural-PDEs-long-time-dynamics.bib  


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
At the continuum level, spatiotemporal physical phenomena such as reaction-diffusion processes and wave propagations can be described by partial differential equations (PDEs). By modeling PDEs, we can understand the complex dynamics of and relationships between parameters across space and time. However, PDEs usually do not have analytical solutions and are often solved numerically using methods such as the finite difference, finite volume, and finite element methods <d-cite key="LoggMardalEtAl2012"></d-cite>. For example, the finite element method (FEM) approximates PDE solutions by first discretizing a sample domain into a mesh of interconnected elements and then solving a system of equations iteratively given a set of boundary conditions, initial conditions, and material properties. 

In this blog, we will show two examples of PDEs, one of which is the Navier-Stokes equation which describes the dynamics of viscous fluids. The equation below shows the 2D Navier-Stokes equation for a viscous and incompressible fluid in vorticity form on a unit torus, where $$w$$ is the vorticity, $$u$$ the velocity field, $$\nu$$ the viscosity coefficient, and $$f(x)$$ is the forcing function. The solution data were from the original paper<d-cite key="li2020fourier"></d-cite> where the problem, with a periodic boundary condition, was solved with a pseudospectral method using a 1e-4 time step with the Crank-Nicolson scheme. 

$$
\begin{gather}
\partial_t w(x, t) + u(x, t) \cdot \nabla w(x, t) = \nu \Delta w(x, t) + f(x), \quad x \in (0,1)^2, t \in [0,T] \\
\nabla \cdot u(x, t) = 0, \quad x \in (0,1)^2, t \in [0,T] \\
w(x, 0) = w_0(x), \quad x \in (0,1)^2
\end{gather}
$$

We can visualize the 2D PDE solution over the 50 time steps:

<div class="l-body-outset" style="display: flex; justify-content: center; align-items: center;">
  <iframe src="{{ 'assets/html/2023-11-05-neural-PDEs-long-time-dynamics/navierstokes.html' | relative_url }}" frameborder="0" scrolling="no" height="600px" width="100%"></iframe>
</div>
<div class="caption">
Solution of 2D Navier-Stokes PDE <d-cite key="li2020fourier"></d-cite> - drag the slider!
</div>

### Motivations for neural PDEs
Well-established numerical methods are very successful in calculating the solutions of PDEs, however, these methods require high computational costs especially for high spatial and temporal resolutions. Furthermore, it is important to have fast and accurate surrogate models that would target problems that require uncertainty quanitifcation, inverse design, and PDE-constrained optimizations. In recent years, there have been growing interests in neural PDE models that act as a surrogate PDE solver, especially neural operators that aim to learn the mapping between input and output solution functions. These models are trained on numerical solutions from existing methods and inferences are orders of magnitude faster than calculating the solutions again through numerical methods. 

In this article, I will first examine if we can apply neural networks to learn the dynamics in PDE solutions and therefore replace PDE solvers with a neural PDE as the surrogate solver. We will start with a base U-Net model with convolutional layers. Next, I will examine the neural operator methods, notably the Fourier Neural Operator (FNO). Primaily, the Fourier neural operator has proven to predict well for PDE solutions and we will use it to compare with the UNet model on the representations learnt in the Fourier layers. Next, I will examine the FNO's performance on another PDE with two dependent states. We will notice that the FNO is capable of learning lower frequency modes but fail to learn local dynamics and higher frequency modes. We then finally introduce some improvements to the FNO to tackle this problem involving local dynamics and long term rollout errors. 


### Dataset and training schemes for the 2D Navier-Stokes PDE
For the dataset, I will start with the 2D time-dependent Navier-Stokes solution ($$\nu$$ = 1e-3) that was shipped from Zongyi Li et al's paper <d-cite key="li2020fourier"></d-cite>. The problem for any given model would then be to learn the mapping from an input solution (vorticity) of t=[0,10] to the solution of t=(10, 40]. For all models involving Navier-Stokes, the original implementations were used, but implementations were improved or new ones were added for the second PDE problem which more details will be shared in later parts of the article. We use 1000 solutions for training and 200 for the test dataset. The models are trained with 500 epochs with an initial learning rate of 0.001, the AdamW optimizer is used with a cosine annealing scheduler. Unless otherwise specified, a relative L2 loss is used for training and prediction of each data batch. For U-Net and FNO2D, the models use 2D convolutions in the spatial domain and recurrently predict through the time domain (autoregressive training). For FNO3D, the time domain is directly used as part of the training. 


## Base model: U-Nets 
Let's begin with examining whether a U-Net with convolutional layers can be used to learn the dynamics. U-Net is a popular model architecture for image to image predictions and image segmentation tasks. It consists a series of downsampling and upsampling layers with skip connections, and my re-implementation is from this repo (https://github.com/khassibi/fourier-neural-operator/blob/main/UNet.py). 

We can use the U-Net to learn the features from the input PDE solution frames and predict the solution in the next time step, treating the 2D solution as an image. As for the time component, the surrogate model takes the input solution from the previous k time steps to predict solution in the next k+1 time step. Then, the solution from the previous k-1 steps are concatenated with the predicted next-step solution as the input back into the model to predict the next step, and so on. In a nutshell, the model is trained to predict autoregressively. 

<div style="text-align: center; margin-right: 10px;"> 
    <div style="width: 50%; margin: auto;"> 
        {% include figure.html path="assets/img/2023-11-05-neural-PDEs-long-time-dynamics/unet_train_test_loss.png" class="img-fluid" %}
    </div>
    <p style="margin-top: 5px;">Training curve for U-Net with average relative L2 train and test loss</p>
</div>

<div style="text-align: center; margin-left: 10px;"> 
    {% include figure.html path="assets/img/2023-11-05-neural-PDEs-long-time-dynamics/unet_2dt_nspred42.gif" class="img-fluid" %}
    <p style="margin-top: 5px;">U-Net's prediction of 2D Navier-Stokes for unseen test set (id=42)</p> 
</div>

The U-Net seems to predict well for the 2D Navier-Stokes test set. However, the average final test loss of 0.0153 is still considerably high. For longer time rollout, the errors can accumulate. Let's examine the FNO2d-t and FNO3d models next.

## Fourier Neural Operators 
Fourier neural operators (FNOs) <d-cite key="li2020fourier"></d-cite> try to learn the mapping between input functions and solution functions <d-cite key="kovachki2021neural"></d-cite>, for example, mapping the solutions from earlier to later time steps for time-dependent PDEs. 

The authors introduced the Fourier layer (SpectralConv2d for FNO2d) which functions as a convolution operator in the Fourier space, and complex weights are optimized in these layers. The input functions are transformed to the frequency domain by performing fast Fourier transforms (torch.fft) and the output functions are then inverse transformed back to the physical space before they are passed through nonlinear activation functions (GeLU) to learn nonlinearity. Fourier transformations are widely used in scientific and engineering applications, such as in signal processing and filtering, where a signal / function is decomposed into its constituent frequencies. In the FNO, the number of Fourier modes is a hyperparameter of the model - the Fourier series up till the Fourier modes are kept (i.e. lower frequency modes are learnt) while higher frequency modes are truncated away. Notably, since the operator kernels are trained in the frequency domain, the model is theoretically capable of predicting solutions that are resolution-invariant. 

### Applying FNO2D and FNO3D on 2D Navier-Stokes time-dependent PDE 
We reimplement and train the FNO2D model on the same train-test data splits for the 2D Navier-Stokes solution. Notably, the final average relative L2 loss (for test set) is 0.00602 after 500 epochs of training. Comparing this with the U-Net that is also trained and predicted with the same scheme, the FNO2D has an improved performance! 

<div style="text-align: center; margin-left: 10px;"> 
    {% include figure.html path="assets/img/2023-11-05-neural-PDEs-long-time-dynamics/2dt_nspred42.gif" class="img-fluid" %}
    <p style="margin-top: 5px;">FNO2D's prediction of 2D Navier-Stokes for unseen test set (id=42)</p> 
</div>

The predicted solutions look impressive and it seems like the dynamics of the multiscale system are learnt well, particularly the global dynamics. Likewise, the FNO3D gives similar results. Instead of just convolutions over the 2D spatial domains, the time-domain is taken in for convolutions in the Fourier space as well. According to the authors, they find that the FNO3D gives better performance than the FNO2D for time-dependent PDEs. However, it uses way more parameters (6560681) compared to FNO2D (928661 parameters) - perhaps the FNO2D with recurrent time is sufficient for most problems.

<div style="text-align: center; margin-right: 10px;"> 
    <div style="width: 50%; margin: auto;"> 
        {% include figure.html path="assets/img/2023-11-05-neural-PDEs-long-time-dynamics/train_test_loss.png" class="img-fluid" %}
    </div>
    <p style="margin-top: 5px;">Training curve for FNO3D with average relative L2 train and test loss</p>
</div>

<div style="text-align: center; margin-left: 10px;"> 
    {% include figure.html path="assets/img/2023-11-05-neural-PDEs-long-time-dynamics/nspred42.gif" class="img-fluid" %}
    <p style="margin-top: 5px;">FNO3D's prediction of 2D Navier-Stokes for unseen test set (id=42)</p> 
</div>


### Representation learning: The learned representations in the Fourier layers 
Next 

{% include figure.html path="assets/img/2023-11-05-neural-PDEs-long-time-dynamics/fourierlayers.png" class="img-fluid" %}

{% include figure.html path="assets/img/2023-11-05-neural-PDEs-long-time-dynamics/convlayers.png" class="img-fluid" %}

How Fourier layers work in contrast with CNN layers, and why they can learn the underlying dynamics regardless of data resolution.

### On the importance of positional encoding 

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


I will first incorporate these techniques with the base FNO model to examine accuracies of long time dynamics. Next, I will examine if attention mechanisms introduced with transformer layers can help improve accuracies for lower frequency modes at longer time scales. These new model architectures would all be compared on the base dataset of Navier-Stokes equations (2D spatially with time dependence). 

### Using ReVIN to normalize and denormalize the time series input for 1D PDEs 


### Improving long time-step rollout with temporal bundling and pushforward tricks 


## Large Kernel Attention 




## Conclusion 
By the end of the project, I hope to propose a new neural PDE approach with **(1) spatial convolutions in the Fourier space, (2) transformer layers for improved attention on local dynamics, and (3) pushforward and temporal bundling tricks to deal with predictions over long time scales**. 