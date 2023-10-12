---
layout: distill
title: Autoregressive Renaissance in Neural PDE Solvers
description: 
  Recent developments in the field of neural partial differential equation (PDE) solvers have placed a strong emphasis on neural operators. However, the paper Message Passing Neural PDE Solver by Brandstetter et al. published in ICLR 2022 revisits autoregressive models and designs a message passing graph neural network that is comparable with or outperforms both the state-of-the-art Fourier Neural Operator and traditional classical PDE solvers in its generalization capabilities and performance. This blog post delves into the key contributions of this work, exploring the strategies used to address the common problem of instability in autoregressive models and the design choices of the message passing graph neural network architecture.
date: 2022-12-01
htmlwidgets: true

# Anonymize when submitting
authors:
  - name: Anonymous

# must be the exact same name as your blogpost
bibliography: 2022-12-01-autoregressive-neural-pde-solver.bib

# Add a table of contents to your post.
#   - make sure that TOC names match the actual section names
#     for hyperlinks within the post to work correctly.
toc:
  - name: Introduction
  - name: Background
    subsections:
    - name: Let's brush up on the basics...
    - name: Solving PDEs the classical way
    - name: Neural Solvers
  - name: Message Passing Neural PDE Solver (MP-PDE)
    subsections:
    - name: The Pushforward Trick and Temporal bundling
    - name: Network Architecture
    - name: Results
  - name: Conclusion

# Below is an example of injecting additional post-specific styles.
# This is used in the 'Layouts' section of this post.
# If you use this post as a template, delete this _styles block.
_styles: >
  .center-screen {
  justify-content: center;
  align-items: center;
  text-align: center;
  }

  .fake-img {
    background: #e2edfc;
    border: 1px solid rgba(0, 0, 0, 0.1);
    border-radius: 25px;
    box-shadow: 0 0px 4px rgba(0, 0, 0, 0.05);
    margin-bottom: 12px;
  }

  .fake-img p {
    font-family: sans-serif;
    color: white;
    margin: 12px 8px;
    text-align: center;
    font-size: 12px;
    line-height: 150%;
  }

  summary {
    color: steelblue
  }

  details[open] {
  --bg: #e2edfc;
  border-radius: 25px;
  padding-left: 8px;
  background: var(--bg);
  outline: 0.5rem solid var(--bg);
  margin: 0 0 2rem 0;
  }

  hr {
    color: #333;
    width:50%;
    margin:0 auto;
    text-align: center;
    height: 2px;
  }

  l-body-outset {
    display: flex;
    justify-content: center;
  }
---
## Introduction
<blockquote>
Improving PDE solvers has trickle down benefits to a vast range of other fields.
</blockquote>

Partial differential equations (PDEs) play a crucial role in modeling complex systems and understanding how they change over time and in space.

They are used across physics and engineering, modeling a wide range of physical phenomena like heat transfer, sound waves, electromagnetism, and fluid dynamics, but they can also be used in finance to model the behavior of financial markets, in biology to model the spread of diseases, and in computer vision to model the processing of images.

They are particularly interesting in deep learning!

<ol>
  <li><span style="color:#9444e2;">Neural networks can be used to solve complex PDEs.</span></li>

  <li><span style="color:#9444e2;">Embedding knowledge of a PDE into a neural network can help it generalize better and/or use less data</span></li>

  <li><span style="color:#9444e2;">PDEs can help explain and/or interpret neural networks.</span></li>
</ol>


Despite their long history, dating back to equations first formalized by Euler over 250 years ago, finding numerical solutions to PDEs continues to be a challenging problem.

The recent advances in machine learning and artificial intelligence have opened up new possibilities for solving PDEs in a more efficient and accurate manner. These developments have the potential to revolutionize many fields, leading to a better understanding of complex systems and the ability to make more informed predictions about their behavior.

The background and problem set up precedes a brief look into classical and neural solvers, and finally discusses the message passing neural PDE solver (MP-PDE) introduced by Brandstetter et al. <d-cite key="brandstetterMessagePassingNeural2022a"></d-cite>.

## Background
### Let\'s brush up on the basics...

*The notation and definitions provided match those in the paper for consistency, unless otherwise specified.*

<div>
<p>
Ordinary differential equations (ODEs) describe how a function changes with respect to a <span style="color:#9444e2">single independent variable</span> and its derivatives. In contrast, PDEs are mathematical equations that describe the behavior of a dependent variable as it changes with respect to <span style="color:#9444e2">multiple independent variables</span> and their derivatives.
</p>
<p>
Formally, for one time dimension and possibly multiple spatial dimensions denoted by \(\textbf{x}=[x_{1},x_{2},x_{3},\text{...}]^{\top} \in \mathbb{X}\), a general (temporal) PDE may be written as
</p>
<p>
$$\partial_{t}\textbf{u}= F\left(t, \textbf{x}, \textbf{u},\partial_{\textbf{x}}\textbf{u},\partial_{\textbf{xx}}\textbf{u},\text{...}\right) \qquad (t,\mathbf{x}) \in [0,T] \times \mathbb{X}$$
</p>
<ul>
  <li>Initial condition:
 \(\mathbf{u}(0,\mathbf{x})=\mathbf{u}^{0}(\mathbf{x})\) for \(\mathbf{x} \in \mathbb{X}\)</li>

  <li>Boundary conditions:
 \(B[ \mathbf{u}](t,x)=0\) for \((t,\mathbf{x}) \in [0,T] \times \partial \mathbb{X}\)</li>
</ul>
</div>

<div class="fake-img l-gutter">
  <p>

  Many equations are solutions to such PDEs alone. For example, the wave equation is given by \(\partial_{tt}u = \partial_{xx}u\). You will find that any function in the form \(u(x,t)=F(x-ct)+\) \(G(x+ct)\) is a potential solution. Initial conditions are used to specify how a PDE "starts" in time, and boundary conditions determine the value of the solution at the boundaries of the region where the PDE is defined.

  </p>
</div>

<details><summary>Types of boundary conditions</summary>
Dirichlet boundary conditions prescribe a fixed value of the solution at a particular point on the boundary of the domain. Neumann boundary conditions, on the other hand, prescribe the rate of change of the solution at a particular point on the boundary. There are also mixed boundary conditions, which involve both Dirichlet and Neumann conditions, and Robin boundary conditions, which involve a linear combination of the solution and its derivatives at the boundary.
</details><br/>

<div class="l-body-outset">
  <iframe src="{{ 'assets/html/2022-12-01-autoregressive-neural-pde-solver/slider.html' | relative_url }}" frameborder='0' scrolling='no' height="750px" width="100%"></iframe>
</div>
<div class="caption">
Example of the wave equation PDE \(\partial^{2}_{t}u = c^{2}\partial^{2}_ {\mathbf{x}}u\). Drag the slider to watch it evolve in time!
</div>

The study of PDEs is in itself split into many broad fields. Briefly, these are two other important properties in addition to the initial and boundary conditions:

<details><summary>Linearity</summary>
<ul>
<li>Linear: the highest power of the unknown function appearing in the equation is one (i.e., a linear combination of the unknown function and its derivatives)</li>
<li>Nonlinear: the highest power of the unknown function appearing in the equation is greater than one</li>
</ul>

</details><br/>

<details><summary>Homogeneity</summary>
<ul>
<li>Homogeneous: PDEs with no constant terms (i.e., the right-hand side is equal to zero)</li>
<li>Inhomogeneous: PDEs with a non-zero constant term on the right-hand side</li>
</ul>

</details><br/>

PDEs can be either linear or nonlinear, homogeneous or inhomogeneous, and can contain a combination of constant coefficients and variable coefficients. They can also involve a variety of boundary conditions, such as Dirichlet, Neumann, and Robin conditions, and can be solved using analytical, numerical, or semi-analytical methods <d-cite key="straussPartialDifferentialEquations2007"></d-cite>.
<hr style="width:40%">

Brandstetter et al. <d-cite key="brandstetterMessagePassingNeural2022a"></d-cite> follow precedence set by Li et al. <d-cite key="liFourierNeuralOperator2021"></d-cite> and Bar-Sinai et al. <d-cite key="bar-sinaiLearningDatadrivenDiscretizations2019"></d-cite>to focus on <span style="color:#9444e2;">PDEs written in conservation form</span>:

<p style="text-align:center;">
\(\partial_{t} \mathbf{u} + \nabla \cdot \mathbf{J}(\mathbf{u}) = 0\)
</p>

<ul>
<li><p>\(J\) is the flux, or the amount of some quantity that is flowing through a region at a given time</p>
</li>
<li><p>\(\nabla \cdot J\) is the divergence of the flux, or the amount of outflow of the flux at a given point</p>
</li>
</ul>


Additionally, they consider <span style="color:#9444e2;">Dirichlet and Neumann</span> boundary conditions.

### Solving PDEs the classical way
A brief search in a library will find numerous books detailing how to solve various types of PDEs.
<!-- Since Brandstetter et al. proposes to numerically solve PDEs, numerical methods are discussed in more detail. -->

<details><summary>Analytical methods: an exact solution to a PDE can be found by mathematical means <d-cite key="straussPartialDifferentialEquations2007"></d-cite>.</summary><br/>

 <ul>
<li>Separation of Variables<ul>
<li>This method involves expressing the solution as the product of functions of each variable, and then solving each function individually. It is mainly used for linear PDEs that can be separated into two or more ordinary differential equations.</li>
</ul>
</li>
<li>Green&#39;s Functions<ul>
<li>This method involves expressing the solution in terms of a Green&#39;s function, which is a particular solution to a homogeneous equation with specified boundary conditions.</li>
</ul>
</li>
</ul>

</details><br/>

<details><summary>Semi-analytical methods: an analytical solution is combined with numerical approximations to find a solution <d-cite key="bartelsNumericalApproximationPartial"></d-cite>.</summary><br/>

<ul>
<li>Perturbation methods<ul>
<li>This method is used when the solution to a PDE is close to a known solution or is a small deviation from a known solution. The solution is found by making a perturbation to the known solution and solving the resulting equation analytically.</li>
</ul>
</li>
<li>Asymptotic methods<ul>
<li>In this method, the solution is represented as a series of terms that are solved analytically. The solution is then approximated by taking the leading terms of the series.</li>
</ul>
</li>
</ul>

</details><br/>

<blockquote>
Very few PDEs have analytical solutions, so numerical methods have been developed to approximate PDE solutions over a wider range of potential problems.
</blockquote>

#### Numerical Methods

Often, approaches for temporal PDEs follow the <span style="color:#9444e2;">method of lines (<abbr title="method of lines">MOL</abbr>)</span>.

Every point of the discretization is then thought of as a separate ODE evolving in time, enabling the use of ODE solvers such as Runge-Kutta methods.

<details><summary>1. Discretizing the problem</summary><br/>

<p>
In the most basic case (<span style="color:#9444e2;">a regular grid</span>), arbitrary spatial and temporal resolutions \(\mathbf{n_{x}}\) and \(n_{t}\) can be chosen and thus used to create a grid where \(\mathbf{n_{x}}\) is a vector containing a resolution for each spatial dimension.
</p>
<hr style="width:40%">
<p>
The domain may also be <span style="color:#9444e2;">irregularly sampled, resulting in a grid-free discretization</span>. This is often the case with real-world data that comes from scattered sensors, for example.
</p>
<p><abbr title="finite difference method">FDM</abbr> or any other time discretization technique can be used to discretize the time domain.
</p>
<p>
One direction of ongoing research seeks to determine discretization methods which can result in more efficient numerical solvers (for example, take larger steps in flatter regions and smaller steps in rapidly changing regions).
</p>

</details><br/>

<details><summary>2. Estimating the spatial derivatives</summary><br/>

<p>
A popular choice when using a gridded discretization is the <span style="color:#9444e2;">finite difference method (FDM)</span>. Spatial derivative operators are replaced by a stencil which indicates how values at a finite set of neighboring grid points are combined to approximate the derivative at a given position. This stencil is based on the Taylor series expansion.
</p>

<p>
{% include figure.html path="assets/img/2022-12-01-autoregressive-neural-pde-solver/fdm_animation.gif" style="max-width:690px;height:auto;" %}
</p>

<div class="caption">
Credits: Augusto Peres, Inductiva <d-cite key="HeatHeatEquation"></d-cite>.
</div>

<hr style="width:40%">
<p>
The <span style="color:#9444e2;">finite volume method (FVM)</span> is another approach which works for irregular geometries. Rather than requiring a grid, the computation domain can be divided into discrete, non-overlapping control volumes used to compute the solution for that portion <d-cite key="bartelsNumericalApproximationPartial"></d-cite>.
</p>

<p>
While this method <span style="color:#9444e2;">only works for conservation form equations</span>, it can handle complex problems with irregular geometries and fluxes that are difficult to handle with other numerical techniques such as the <abbr title="finite difference method">FDM</abbr>.
</p>
<hr style="width:40%">
<p>
In the <span style="color:#9444e2;">pseudospectral method (PSM)</span>, PDEs are solved pointwise in physical space by using basis functions to approximate the spatial derivatives <d-cite key="brandstetterMessagePassingNeural2022a"></d-cite>. 
</p>

<p>
It is well-suited for solving problems with <span style="color:#9444e2;">smooth solutions and periodic boundary conditions</span>, but its performance drops for irregular or non-smooth solutions.
</p>
</details><br/>

<details><summary>3. Time updates</summary><br/>

The resulting problem is a set of temporal ODEs which can be solved with classical ODE solvers such as any member of the Runge-Kutta method family.

</details><br/>

#### Limitations of Classical Methods

The properties of a PDE, such as its order, linearity, homogeneity, and boundary conditions, determine its solution method. <span style="color:#9444e2;">Different methods have been developed based on the different properties and requirements of the problem at hand.</span> Brandstetter at al. categorizes these requirements into the following <d-cite key="brandstetterMessagePassingNeural2022a"></d-cite>:

<div>

<table>
<thead>
<tr>
<th>User</th>
<th>Structural</th>
<th>Implementational</th>
</tr>
</thead>
<tbody>
<tr>
<td>Computation efficiency, computational cost, accuracy, guarantees (or uncertainty estimates), generalization across PDEs</td>
<td>Spatial and temporal resolution, boundary conditions, domain sampling regularity, dimensionality</td>
<td>Stability over long rollouts, preservation of invariants</td>
</tr>
</tbody>
</table>

<p>
The countless combinations of requirements resulted in what Bartels defines as a <span style="color:#9444e2;">splitter field</span> <d-cite key="bartelsNumericalApproximationPartial"></d-cite>: a specialized classical solver is developed for each sub-problems, resulting in many specialized tools rather than a single one.
</p>

<p>
These methods, while effective and mathematically proven, often come at high computation costs. Taking into account that PDEs often exhibit chaotic behaviour and are sensitive to any changes in their parameters, <span style="color:#ff4f4b;">re-running a solver every time a coefficient or boundary condition changes in a single PDE can be computationally expensive</span>.
</p>

</div>

<div class="fake-img l-gutter">
  <p>

  Courant–Friedrichs–Lewy (CFL) condition

  </p>
  <p>
  <it>The maximum time step size should be proportional to the minimum spatial grid size.</it>
  </p>
  <p>
  According to this condition, as the number of dimensions increases, the size of the temporal step must decrease and therefore numerical solvers become very slow for complex PDEs.
  </p>
</div>


### Neural Solvers
<p>
Neural solvers offer some very desirable properties that may serve to unify some of this splitter field. Neural networks can <span style="color:#9444e2;">learn and generalize to new contexts</span> such as different initial/boundary conditions, coefficients, or even different PDEs entirely <d-cite key="brandstetterMessagePassingNeural2022a"></d-cite>. They can also circumvent the CFL condition, making them a promising avenue for solving highly complex PDEs such as those found in weather prediction.
</p>

#### Neural operator methods

Neural operator methods <span style="color:#9444e2;">model the solution of a PDE as an operator that maps inputs to outputs</span>. The problem is set such that a neural operator $$\mathcal{M}$$ satisfies $$\mathcal{M}(t,\mathbf{u}^{0}) = \mathbf{u}(t)$$ where $$\mathbf{u}^{0}$$ are the initial conditions <d-cite key="luDeepONetLearningNonlinear2021, brandstetterMessagePassingNeural2022a"></d-cite>.

<p>
One of the current state-of-the-art models is the <span style="color:#9444e2;">Fourier Neural Operator (FNO)</span> <d-cite key="liFourierNeuralOperator2021"></d-cite>. It operates within Fourier space and takes advantage of the convolution theorem to place the integral kernel in Fourier space as a convolutional operator.
</p>
<div>
  <p>
  These global integral operators (implemented as Fourier space convolutional operators) are combined with local nonlinear activation functions, resulting in an architecture which is <span style="color:#9444e2;">highly expressive yet computationally efficient, as well as being resolution-invariant</span>.
  </p>
  <p>
  While the vanilla <abbr title="Fourier neural operator">FNO</abbr> required the input function to be defined on a grid due to its reliance on the FFT, further work developed mesh-independent variations as well <d-cite key="kovachkiNeuralOperatorLearning2022"></d-cite>.
  </p>
</div>

<div class="fake-img l-gutter">
  <p>

  Convolution Theorem

  </p>
  <p>
  The Fourier transform of the convolution of two signals is equal to the pointwise product of their individual Fourier transforms
  </p>
</div>

<p>
{% include figure.html path="assets/img/2022-12-01-autoregressive-neural-pde-solver/FNO.png" style="max-width:80%;height:auto;" %}
</p>
<div class="caption">
<abbr title="Fourier neural operator">FNO</abbr> architecture. For more details, see <a href="https://zongyi-li.github.io/blog/2020/fourier-pde/">this blogpost</a>. Credits: Li et al. <d-cite key="liFourierNeuralOperator2021"></d-cite>.
</div>

<p>
Neural operators are able to operate on multiple domains and can be completely data-driven.
</p>
<p>
However, these models <span style="color:#ff4f4b;">do not tend to predict out-of-distribution \(t\)</span> and are therefore limited when dealing with temporal PDEs. Another major barrier is their relative <span style="color:#ff4f4b;">lack of interpretability and guarantees</span> compared to classical solvers. 
</p>

#### Autoregressive methods

While neural operator methods directly mapped inputs to outputs, <span style="color:#9444e2;">autoregressive methods take an iterative approach instead</span>. For example, iterating over time results in a problem such as $$\mathbf{u}(t+\Delta t) = \mathcal{A}(\Delta t, \mathbf{u}(t))$$ where $$\mathcal{A}$$ is some temporal update <d-cite key="brandstetterMessagePassingNeural2022a"></d-cite>.

<div class="l-body-outset">
<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/2022-12-01-autoregressive-neural-pde-solver/rnn.png" class="img-fluid rounded" %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/2022-12-01-autoregressive-neural-pde-solver/wavenet.gif" class="img-fluid rounded" %}
    </div>
</div>
<div class="caption">
    Similarly to <abbr title="recurrent neural networks">RNN</abbr>s (left), autoregressive models take previous time steps to predict the next time step. However, autoregressive models (right) are entirely feed-forward and take the previous predictions as inputs rather than storing them in some hidden state. Credits: RNN diagram from Colah's Blog <d-cite key="UnderstandingLSTMNetworks"></d-cite>, WaveNet from Deepmind Blog <d-cite key="WaveNetGenerativeModel"></d-cite>
</div>
</div>

Three autoregressive works mentioned by Brandstetter et al. are hybrid methods which use neural networks to predict certain parameters for finite volume, multigrid, and iterative finite elements methods. <span style="color:#9444e2;">All three retain a (classical) computation grid which makes them somewhat interpretable</span> <d-cite key="bar-sinaiLearningDatadrivenDiscretizations2019, greenfeldLearningOptimizeMultigrid2019a, hsiehLearningNeuralPDE2019"></d-cite>.

<div class="fake-img l-gutter">
  <p>
  Other autoregressive models include PixelCNN for images, WaveNet for audio, and the Transformer for text.
  </p>
</div>

However, autoregressive models have not gained the acclaim seen by neural operators as a whole.

This is on one hand due to their <span style="color:#ff4f4b;">limitations in generalization</span>. In Hsieh et al.'s case, an existing numerical method must be used to craft a complementary neural iterator <d-cite key="hsiehLearningNeuralPDE2019"></d-cite>.

Another major concern is the <span style="color:#ff4f4b;">accumulation of error</span>, which is particularly detrimental for PDE problems that often exhibit chaotic behavior <d-cite key="brandstetterMessagePassingNeural2022a"></d-cite>.

## Message Passing Neural PDE Solver (MP-PDE)

Brandstetter et al. propose a <span style="color:#9444e2;">fully neural PDE solver which capitalizes on neural message passing</span>. The overall architecture is laid out below, consisting of an <abbr title="multilayer perceptron">MLP</abbr> encoder, a <abbr title="graph neural network">GNN</abbr> processor, and a CNN decoder.

<div class="l-body-outset">
{% include figure.html path="assets/img/2022-12-01-autoregressive-neural-pde-solver/MP-PDE-Solver.png" style="max-width:100%;height:auto;" %}
<div class="caption">
Overall MP-PDE architecture. Credits: Brandstetter et al. <d-cite key="brandstetterMessagePassingNeural2022a"></d-cite>.
</div>
</div>

At its core, this model is autoregressive and thus faces the same challenge listed above. Two key contributions of this work are the <span style="color:#9444e2;">pushforward trick and temporal bundling which mitigate the potential butterfly effect of error accumulation</span><d-cite key="brandstetterMessagePassingNeural2022a"></d-cite>. The network itself, being fully neural, is capable of generalization across many changes as well.

### The Pushforward Trick and Temporal Bundling

<div class="l-body-outset">
{% include figure.html path="assets/img/2022-12-01-autoregressive-neural-pde-solver/pushforward3.jpg" style="max-width:100%;height:auto;" %}
<div class="caption">
Pushforward trick compared to one-step and unrolled training. Credits: Brandstetter et al. <d-cite key="brandstetterMessagePassingNeural2022a"></d-cite>.
</div>
</div>

During testing, the model uses current time steps (first from data, then <span style="color:#9444e2;">from its own predictions</span>) to approximate the next time step.

This results in a distribution shift problem because the inputs are no longer solely from ground truth data: <span style="color:#9444e2;">the distribution learned during training will always be an approximation of the true data distribution</span>. The model will appear to overfit to the one-step training distribution and perform poorly the further it continues to predict.

An adversarial-style stability loss is added to the one-step loss so that the training distribution is brought closer to the test time distribution <d-cite key="brandstetterMessagePassingNeural2022a"></d-cite>:

<details><summary style="text-align:center;color:black;">
\(L_{\text{one-step}} =\) <span style="color:#23a15c;">\(\mathbb{E}_{k}\)</span> <span style="color:#928b54;">\(\mathbb{E}_{\mathbf{u^{k+1}|\mathbf{u^{k},\mathbf{u^{k} \sim p_{k}}}}}\)</span> \([\) <span style="color:#5588e0;">\(\mathcal{L}\)</span> \((\) <span style="color:#9444e2;">\(\mathcal{A}(\mathbf{u}^{k})\)</span> \(,\) <span style="color:#46b4af;">\(\mathbf{u}^{k+1}]\)</span>
</summary>

<p>
The <span style="color:#5588e0;">loss function</span> is used to evaluate the difference between the <span style="color:#9444e2;">temporal update</span> and the <span style="color:#46b4af;">expected next state</span>, and the overall one-step loss is calculated as the expected value of this loss over <span style="color:#23a15c;">all time-steps</span> and <span style="color:#928b54;">all possible next states</span>.
</p>
</details><br style="line-height:5px"/>

<p style="text-align:center;">
\(L_{\text{stability}} = \mathbb{E}_{k}\mathbb{E}_{\mathbf{u^{k+1}|\mathbf{u^{k},\mathbf{u^{k} \sim p_{k}}}}}[\mathbb{E}_{\epsilon | \mathbf{u}^{k}} [\mathcal{L}(\mathcal{A}(\mathbf{u}^{k}+\) <span style="color:#faad18;">\(\epsilon\)</span> \()),\mathbf{u}^{k+1}]]\)
</p>

<p style="text-align:center;">
\(L_{\text{total}} = L_{\text{one-step}} + L_{\text{stability}}\)
</p>

<p>
The stability loss is largely based off the one-step loss, but now assumes that the temporal update uses <span style="color:#faad18;">noisy data</span>.
</p>

<p>
The pushforward trick lies in the choice of <span style="color:#faad18;">\(\epsilon\)</span> such that \(\mathbf{u}^{k}+\epsilon = \mathcal{A}(\mathbf{u}^{k-1})\), similar to the test time distribution. Practically, it is implemented to be <span style="color:#9444e2;">noise from the network itself</span> so that as the network improves, the loss decreases.
</p>

<p>
Necessarily, the noise of the network must be known or calculated to implement this loss term. So, <span style="color:#9444e2;">the model is unrolled for 2 steps</span> but only backpropagated over the most recent unroll step, which already has the neural network noise <d-cite key="brandstetterMessagePassingNeural2022a"></d-cite>.
</p>

<p>
While the network could be unrolled during training, this not only slows the training down but also might result in the network learning shortcuts across unrolled steps.
</p>

**Temporal bundling**

<div class="row mt-3">
    <div class="col-8">
        {% include figure.html path="assets/img/2022-12-01-autoregressive-neural-pde-solver/NN-AR.jpg" class="img-fluid rounded" %}
    </div>
    <div class="col-4">
        {% include figure.html path="assets/img/2022-12-01-autoregressive-neural-pde-solver/temporalbundling.jpg" class="img-fluid rounded" %}
    </div>
</div>
<div class="caption">
    Temporal bundling compared to neural operators and autoregressive models. Credits: Brandstetter et al. <d-cite key="brandstetterMessagePassingNeural2022a"></d-cite>.
</div>
<!-- </div> -->

This trick complements the previous by <span style="color:#9444e2;">reducing the amount of times the test time distribution changes</span>. Rather than predicting a single value at a time, the MP-PDE predicts multiple time-steps at a time, as seen above <d-cite key="brandstetterMessagePassingNeural2022a"></d-cite>.

### Network Architecture

<abbr title="graph neural network">GNN</abbr>s have been used as PDE solvers in a variety of works <d-cite key="liNeuralOperatorGraph2020, eliasofPdegcnNovelArchitectures2021, iakovlevLearningContinuoustimePDEs2021"></d-cite>; however, in this implementation, <span style="color:#9444e2;">links can be drawn directly from the <abbr title="method of lines">MOL</abbr> to each component of the network architecture centering around the use of a message passing algorithm.

<table>
<thead>
<tr>
<th>Classical Numerical Method</th>
<th>MP-PDE Network Component</th>
</tr>
</thead>
<tbody>
<tr>
<td>Partitioning the problem onto a grid</td>
<td>Encoder <br /><em>Encodes a vector of solutions into node embeddings</em></td>
</tr>
<tr>
<td>Estimating the spatial derivatives</td>
<td>Processor <br /><em>Estimates spatial derivatives via message passing</em></td>
</tr>
<tr>
<td>Time updates</td>
<td>Decoder <br /><em>Combines some representation of spatial derivatives smoothed into a time update</em></td>
</tr>
</tbody>
</table>

<ol>
<li>Encoder
<p>
The encoder is implemented as a two-layer <abbr title="multilayer perceptron">MLP</abbr> which computes an embedding for each node \(i\) to cast the data to a <span style="color:#9444e2;">non-regular integration grid</span>:
</p>
<details><summary style="text-align:center;color:black;">\(\mathbf{f}_{i}^{0} = \epsilon^{v}([\mathbf{u}_{i}^{k-K:k},\mathbf{x}_{i},t_{k},\theta_{PDE}])\)
</summary>

where \(\mathbf{u}_{i}^{k-K:k}\) is a vector of previous solutions (the length equaling the temporal bundle length), \(\mathbf{x}_{i}\) is the node's position, \(t_{k}\) is the current timestep, and \(\theta_{PDE}\) holds equation parameters.

</details>
</li>
<li>
Processor

<p>
The node embeddings from the encoder are then used in a message passing <abbr title="graph neural network">GNN</abbr>. <a id="spatialderivative" style="text-decoration:none;">The message passing algorithm, which approximates spatial derivatives, is run \(M\) steps using the following updates:</a>
</p>

<details><summary style="text-align:center;color:black;">

\(\text{edge } j \to i \text{ message:} \qquad \mathbf{m}_{ij}^{m} =\) <span style="color:#ae46b4;">\(\phi\)</span> \((\) <span style="color:#b4a546;">\(\mathbf{f}_{i}^{m}, \mathbf{f}_{j}^{m},\)</span>  <span style="color:steelblue;">\(\mathbf{u}_{i}^{k-K:k}-\mathbf{u}_{j}^{k-K:k}\)</span>, <span style="color:#6546b4;">\(\mathbf{x}_{i}-\mathbf{x}_{j}\)</span>, <span style="color:#46b4af;">\(\theta_{PDE}\)</span> \())\)

</summary>

The <span style="color:#6546b4;">difference in spatial coordinates</span> helps enforce translational symmetry and, combined with the <span style="color:steelblue;">difference in node solutions</span>, relates the message passing to a local difference operator. The addition of the <span style="color:#46b4af;">PDE parameters</span> is motivated by considering what the MP-PDE should generalize over: by adding this information in multiple places, flexibility can potentially be learned since all this information (as well as the <span style="color:#b4a546;">node embeddings</span>) is fed through <span style="color:#ae46b4;">a two-layer <abbr title="multilayer perceptron">MLP</abbr></span>.

In addition, the solution of a PDE at any timestep must respect the boundary condition (the same as in classical methods for BVPs), so adding the <span style="color:#46b4af;">PDE parameters</span> in the edge update provides knowledge of the boundary conditions to the neural solver.

</details>

<br/>

<details><summary style="text-align:center;color:black;">

\(\text{node } i \text{ update:} \qquad\) <span style="color:#ff4f4b;">\(\mathbf{f}_{i}^{m+1}\)</span> \(=\) <span style="color:#928b54;">\(\psi\)</span> \((\) <span style="color:#5588e0;">\(\mathbf{f}^{m}_{i}\)</span>, <span style="color:#722e4e;">\(\sum_{j \in \mathcal{N}(i)} \mathbf{m}_{ij}^{m}\)</span>, <span style="color:#46b4af;">\(\theta_{PDE}\)</span> \()\)

</summary>

The <span style="color:#ff4f4b;">future node embedding</span> is updated using <span style="color:#5588e0;">the current node embedding</span>, <span style="color:#722e4e;">the aggregation of all received messages</span>, and (again) the <span style="color:#46b4af;">PDE parameters</span>. This information is also fed through <span style="color:#928b54;">a two-layer <abbr title="multilayer perceptron">MLP</abbr></span>.

</details><br/>

<p>
Bar-Sinai et al. explores the relationship between <abbr title="finite difference method">FDM</abbr> and <abbr title="finite volume method">FVM</abbr> as used in the method of lines <d-cite key="bar-sinaiLearningDatadrivenDiscretizations2019"></d-cite>. In both methods, the \(n^{th}\) order derivative at a point \(x\) is approximated by
</p>

<p style="text-align:center;">
\(\partial^{(n)}_{x}u \approx \sum_{i} a^{(n)}_{i} u_{i}\)
</p>

<p>
for some precomputed coefficients \(a^{(n)}_{i}\). <span style="color:#9444e2;">The right hand side parallels the message passing scheme</span>, which aggregates the local difference (<span style="color:steelblue;">\(\mathbf{u}_{i}^{k-K:k}-\mathbf{u}_{j}^{k-K:k}\)</span> in the edge update) and other (learned) embeddings over neighborhoods of nodes. 
</p>

<p>
This relationship gives an intuitive understanding of the message passing <abbr title="graph neural network">GNN</abbr>, which mimics <abbr title="finite difference method">FDM</abbr> for a single layer, <abbr title="finite volume method">FVM</abbr> for two layers, and <abbr title="Weighted Essentially Non-Oscillatory (5th order)">WENO5</abbr> for three layers <d-cite key="brandstetterMessagePassingNeural2022a"></d-cite>. <abbr title="Weighted Essentially Non-Oscillatory (5th order)">WENO5</abbr> is a numerical interpolation scheme used to reconstruct the solution at cell interfaces in <abbr title="finite volume method">FVM</abbr>.
</p>

<p>
While the interpretation is desirable, how far this holds in the actual function of the <abbr title="message passing graph neural network">MP-GNN</abbr> is harder to address. The concepts of the nodes as integration points and messages as local differences break down as the nodes and edges update. In addition, the furthest node that contributes a message from for any point is at \(n\) edges away for the \(n^{th}\) layer (or a specified limit). This results in a very coarse and potentially underinformed approximation for the first layer which is then propagated to the next layers. However, both the updates use two layer <abbr title="multilayer perceptron">MLP</abbr>s which (although abstracting away from their respective interpretations) may in effect learn optimal weightings to counterbalance this.
</p>
</li>
<li>
Decoder

<p>
The approximated spatial derivatives are then <span style="color:#9444e2;">combined and smoothed using a CNN</span> which outputs a bundle of next time steps (recall temporal bundling) \(\mathbf{d}_{i}\). The solution is then updated:
</p>

<p style="text-align:center;">
\(\mathbf{u}^{k+l}_{i} = u^{k}_{i} + (t_{k+l}-t_{k})\mathbf{d}^{l}_{i}\)
</p>

<p>
Some precedence is seen, for example, in classical linear multistep methods which (though effective) face stability concerns. Since the CNN is adaptive, it appears that it avoids this issue <d-cite key="brandstetterMessagePassingNeural2022a"></d-cite>.
</p>
</li>
</ol>

### Results

<details><summary>Quantitative measures: accumulated error, runtime</summary>
<p>
Accumulated error: \(\frac{1}{n_{x}} \sum_{x,t} MSE\)
</p>
<p>
Runtime (s): Measured time taken to run for a given number of steps.
</p>

</details>

<blockquote>
As a general neural PDE solver, the <abbr title="message passing graph neural network">MP-GNN</abbr> surpasses even the current state-of-the-art <abbr title="Fourier neural operator">FNO</abbr>.
</blockquote>

For example, after training a neural model and setting up an instance of <abbr title="method of lines">MOL</abbr>, this is a brief comparison of how they can generalize without re-training.

<table>
<thead>
<tr>
<th>Generalization to...</th>
<th><abbr title="message passing graph neural network">MP-GNN</abbr></th>
<th><abbr title="Fourier neural operator">FNO</abbr></th>
<th>Classical (<abbr title="method of lines">MOL</abbr>)</th>
</tr>
</thead>
<tbody>
<tr>
<td>New PDEs</td>
<td>Yes</td>
<td>No</td>
<td>No</td>
</tr>
<tr>
<td>Different resolutions</td>
<td>Yes</td>
<td>Yes</td>
<td>No (unless downsampling)</td>
</tr>
<tr>
<td>Changes in PDE parameters</td>
<td>Yes</td>
<td>Yes</td>
<td>Sometimes</td>
</tr>
<tr>
<td>Non-regular grids</td>
<td>Yes</td>
<td>Some</td>
<td>Yes (dependent on implementation)</td>
</tr>
<tr>
<td>Higher dimensions</td>
<td>Yes</td>
<td>No</td>
<td>No</td>
</tr>
</tbody>
</table>

<div class="l-body-outset">
{% include figure.html path="assets/img/2022-12-01-autoregressive-neural-pde-solver/shock_formation.png" style="max-width:100%;height:auto;" %}
<div class="caption">
Demonstration of shock formation using MP-PDE from different training data resolutions. Credits: Brandstetter et al. <d-cite key="brandstetterMessagePassingNeural2022a"></d-cite>.
</div>
</div>

This experiment exemplifies the MP-PDE's ability to model shocks (where both the <abbr title="finite difference method">FDM</abbr> and PSM methods fail) across multiple resolutions. Even at a fifth of the resolution of the ground truth, both the small and large shocks are captured well.

<div class="l-body-outset">
{% include figure.html path="assets/img/2022-12-01-autoregressive-neural-pde-solver/2dshock.jpg" style="max-width:100%;height:auto;" %}
<div class="caption">
Demonstration of shock formation using MP-PDE from different training data resolutions. Credits: Brandstetter et al. <d-cite key="brandstetterMessagePassingNeural2022a"></d-cite>.
</div>
</div>

The same data is displayed in 2D to show the time evolution. After about 7.5s, the error accumulation is large enough to visibly diverge from the ground truth. The predictions become unreliable due to error accumulation.

In practice, this survival time should be empirically found (as seen here) to determine for how long the solution is reliable. However, the ground truth would be needed for comparison, rendering this as another chicken-egg problem.

<table>
<thead>
  <tr>
    <th colspan="2"></th>
    <th colspan="4" style="border-left:1px solid lightgrey;">Accumulated Error</th>
    <th colspan="2" style="border-left:1px solid lightgrey;">Runtime [s]</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td colspan="2">
    \(\quad (n_{t},n_{x})\)
    </td>
    <td style="border-left:1px solid lightgrey;">WENO5</td>
    <td>FNO-RNN</td>
    <td style="border-left:1px solid lightgrey;">FNO-PF</td>
    <td>MP-PDE</td>
    <td style="border-left:1px solid lightgrey;">WENO5</td>
    <td>MP-PDE</td>
  </tr>
  <tr>
    <td><b>E1</b></td>
    <td>(250,100)</td>
    <td style="border-left:1px solid lightgrey;">2.02</td>
    <td>11.93</td>
    <td style="border-left:1px solid lightgrey;">0.54</td>
    <td>1.55</td>
    <td style="border-left:1px solid lightgrey;">1.9</td>
    <td>0.09</td>
  </tr>
  <tr>
    <td><b>E1</b></td>
    <td>(250, 50)</td>
    <td style="border-left:1px solid lightgrey;">6.23</td>
    <td>29.98</td>
    <td style="border-left:1px solid lightgrey;">0.51</td>
    <td>1.67</td>
    <td style="border-left:1px solid lightgrey;">1.8</td>
    <td>0.08</td>
  </tr>
  <tr>
    <td><b>E1</b></td>
    <td>(250, 40)</td>
    <td style="border-left:1px solid lightgrey;">9.63</td>
    <td>10.44</td>
    <td style="border-left:1px solid lightgrey;">0.57</td>
    <td>1.47</td>
    <td style="border-left:1px solid lightgrey;">1.7</td>
    <td>0.08</td>
  </tr>
  <tr>
    <td><b>E2</b></td>
    <td>(250, 100)</td>
    <td style="border-left:1px solid lightgrey;">1.19</td>
    <td>17.09</td>
    <td style="border-left:1px solid lightgrey;">2.53</td>
    <td>1.58</td>
    <td style="border-left:1px solid lightgrey;">1.9</td>
    <td>0.09</td>
  </tr>
  <tr>
    <td><b>E2</b></td>
    <td>(250, 50)</td>
    <td style="border-left:1px solid lightgrey;">5.35</td>
    <td>3.57</td>
    <td style="border-left:1px solid lightgrey;">2.27</td>
    <td>1.63</td>
    <td style="border-left:1px solid lightgrey;">1.8</td>
    <td>0.09</td>
  </tr>
  <tr>
    <td><b>E2</b></td>
    <td>(250, 40)</td>
    <td style="border-left:1px solid lightgrey;">8.05</td>
    <td>3.26</td>
    <td style="border-left:1px solid lightgrey;">2.38</td>
    <td>1.45</td>
    <td style="border-left:1px solid lightgrey;">1.7</td>
    <td>0.08</td>
  </tr>
  <tr>
    <td><b>E3</b></td>
    <td>(250, 100)</td>
    <td style="border-left:1px solid lightgrey;">4.71</td>
    <td>10.16</td>
    <td style="border-left:1px solid lightgrey;">5.69</td>
    <td>4.26</td>
    <td style="border-left:1px solid lightgrey;">4.8</td>
    <td>0.09</td>
  </tr>
  <tr>
    <td><b>E3</b></td>
    <td>(250, 50)</td>
    <td style="border-left:1px solid lightgrey;">11.71</td>
    <td>14.49</td>
    <td style="border-left:1px solid lightgrey;">5.39</td>
    <td>3.74</td>
    <td style="border-left:1px solid lightgrey;">4.5</td>
    <td>0.09</td>
  </tr>
  <tr>
    <td><b>E3</b></td>
    <td>(250, 40)</td>
    <td style="border-left:1px solid lightgrey;">15.97</td>
    <td>20.90</td>
    <td style="border-left:1px solid lightgrey;">5.98</td>
    <td>3.70</td>
    <td style="border-left:1px solid lightgrey;">4.4</td>
    <td>0.09</td>
  </tr>
</tbody>
</table>
<div class="caption">
Table of experiment results adapted from paper. Credits: Brandstetter et al. <d-cite key="brandstetterMessagePassingNeural2022a"></d-cite>.
</div>

<details><summary>Abbreviations</summary>

<table>
<thead>
<tr>
<th>Shorthand</th>
<th>Meaning</th>
</tr>
</thead>
<tbody>
<tr>
<td><strong>E1</strong></td>
<td>Burgers&#39; equation without diffusion</td>
</tr>
<tr>
<td><strong>E2</strong></td>
<td>Burgers&#39; equation with variable diffusion</td>
</tr>
<tr>
<td><strong>E3</strong></td>
<td>Mixed equation, see below</td>
</tr>
<tr>
<td>\(n_{t}\)</td>
<td>Temporal resolution</td>
</tr>
<tr>
<td>\(n_{x}\)</td>
<td>Spatial resolution</td>
</tr>
<tr>
<td>WENO5</td>
<td>Weighted Essentially Non-Oscillatory (5th order)</td>
</tr>
<tr>
<td><abbr title="Fourier neural operator">FNO</abbr>-<abbr title="recurrent neural networks">RNN</abbr></td>
<td>Recurrent variation of <abbr title="Fourier neural operator">FNO</abbr> from original paper</td>
</tr>
<tr>
<td><abbr title="Fourier neural operator">FNO</abbr>-PF</td>
<td><abbr title="Fourier neural operator">FNO</abbr> with the pushforward trick added</td>
</tr>
<tr>
<td>MP-PDE</td>
<td>Message passing neural PDE solver</td>
</tr>
</tbody>
</table>

<p>
The authors form a general PDE in the form
</p>

<p style="text-align:center;">
\([\partial_{t}u + \partial_{x}(\alpha u^{2} - \beta \partial_{x} u + \gamma \partial_{xx} u)](t,x) = \delta (t,x)\)
</p>

<p style="text-align:center;">
\(u(0,x) = \delta(0,x)\)
</p>

<p>
such that \(\theta_{PDE} = (\alpha, \beta, \gamma)\) and different combinations of these result in the heat equation, Burgers' equation, and the KdV equation. \(\delta\) is a forcing term, allowing for greater variation in the equations being tested.
</p>

</details>

For this same experiment, the error and runtimes were recorded when solving using <abbr title="Weighted Essentially Non-Oscillatory (5th order)">WENO5</abbr>, the recurrent variant of the <abbr title="Fourier neural operator">FNO</abbr> (<abbr title="Fourier neural operator">FNO</abbr>-<abbr title="recurrent neural networks">RNN</abbr>), the <abbr title="Fourier neural operator">FNO</abbr> with the pushforward trick (<abbr title="Fourier neural operator">FNO</abbr>-PF), and the MP-PDE.

<blockquote>
The pushforward trick is successful in mitigating error accumulation.
</blockquote>
Comparing the accumulated errors of <abbr title="Fourier neural operator">FNO</abbr>-<abbr title="recurrent neural networks">RNN</abbr> and the <abbr title="Fourier neural operator">FNO</abbr>-PF across all experiments highlights the advantage of the pushforward trick. While the MP-PDE outperforms all other tested methods in the two generalization experiments **E2** and **E3**, the <abbr title="Fourier neural operator">FNO</abbr>-PF is most accurate for **E1**.

When solving a single equation, the <abbr title="Fourier neural operator">FNO</abbr> likely performs better, though both <abbr title="Fourier neural operator">FNO</abbr>-PF and MP-PDE methods outperform <abbr title="Weighted Essentially Non-Oscillatory (5th order)">WENO5</abbr>.

<blockquote>
Neural solvers are resolution-invariant.
</blockquote>
As $$n_{x}$$ is decreased, <abbr title="Weighted Essentially Non-Oscillatory (5th order)">WENO5</abbr> performs increasingly worse whereas all the neural solvers remain relatively stable.
<blockquote>
Neural solver runtimes are constant to resolution.
</blockquote>
Additionally, the runtimes of <abbr title="Weighted Essentially Non-Oscillatory (5th order)">WENO5</abbr> decrease (likely proportionally) since fewer steps require fewer calculations, but the MP-PDE runtimes again appear relatively stable.

## Conclusion

#### Future Directions

The authors conclude by discussing some future directions.

For example, the MP-PDE can be modified for <span style="color:#9444e2;">PDE *retrieval* (which they call parameter optimization)</span>. There is some precedence for this: Cranmer et al. develop a method which fits a symbolic regression model (eg.: PySR, eureqa) to the learned internal functions of a GNN <d-cite key="cranmerDiscoveringSymbolicModels2020"></d-cite>. Alternatively, the MP-PDE's capacity for generalization means that biasing the model with a prior to determine coefficients could be as simple as training on an example instance of the predicted equation, fitting this model on real world data (much like a finetuning process), and extracting the $$\theta_{PDE}$$ parameters.

<span style="color:#9444e2;">Adaptive time stepping</span> is another avenue which could make the model more efficient and accurate by taking large steps over stable/predictable solution regions and smaller steps over changing/unpredictable solution regions. The choice of a CNN for the decoder works well over regular inputs and outputs, but other options like attention-based architectures could potentially weigh the outputted node embeddings such that the model might learn different time steps. Some care would have to be taken with temporal bundling in this case, since the resulting vectors would be potentially irregular in time.

The one-step loss which is the basis of the <span style="color:#9444e2;">adversarial-style loss</span> is also used in reinforcement learning, which frequently uses deep autoregressive models. Other formulations which borrow from reinforcement learning (where distribution shifts are quite common) and other fields could prove successful as well. Transformer-based natural language processing are now capable of capturing extremely long sequence dependencies and generating coherent long-form text. Since [Transformers are GNNs](https://graphdeeplearning.github.io/post/transformers-are-gnns/) which use attention to aggregate neighborhoods, this may be a viable avenue to explore.

**One more potential direction is inspired by the recent GRAND paper.**

Brandstetter et al. emphasizes the value of relationships to classical solvers - in fact, this is one of the key benefits of hybrid autoregressive models. However, modeling continuous functions as in neural operator models typically outperforms their competitors. Even the MP-PDE is fully neural, making it less explanable than the hybrid autoregressive models introduced earlier.

The Graph Neural Diffusion (GRAND) model introduced by Chamberlain et al. demonstrates that <span style="color:#9444e2;"><abbr title="graph neural network">GNN</abbr> can be crafted using differential equations</span> (like diffusion processes) where, <a href="#spatialderivative">similarly to Brandstetter et al.</a>, the spatial derivative is analogous to the difference between node features <d-cite key="chamberlainGRANDGraphNeural2021a"></d-cite>. The layers are however analogous to the temporal change in a continuous-time differential equation, diverging from the MP-PDE intuition.

Rather than "representationally [containing] some classical methods" <d-cite key="brandstetterMessagePassingNeural2022a"></d-cite>, GRAND provides a <span style="color:#9444e2;">mathematical framework</span> which not only offers explanability, but also a method to design new architectures with theoretical guarantees like stability or convergence <d-cite key="chamberlainGRANDGraphNeural2021a"></d-cite>.

For example, standard <abbr title="message passing graph neural network">MP-GNN</abbr>s are shown to be equivalent to the explicit single-step Euler scheme; other classical solvers result in different flavours of message passing. Using GRAND to extend the MP-PDE would require rethinking the encoder and decoder, but the potential benefit could result in more reliability and therefore wider adoption of neural solvers for real world applications.

#### Remarks

In their paper "Message Passing Neural PDE Solver", Brandstetter at al. present a well-motivated neural solver based on the principle of message passing. The key contributions are the end-to-end network capable of one-shot generalization, and the mitigation of error accumulation in autoregressive models via temporal bundling and the pushforward trick. Note that the latter are self-contained can be applied to other architectures (as in the FNO-PF), providing a valuable tool to improve autoregressive models.