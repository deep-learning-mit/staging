---
layout: distill
title: (Proposal) Physics-informed Learning for Chaotic Dynamics Prediction
description: Project proposal submission by Sunbochen Tang.
date: 2023-11-10
htmlwidgets: true

# Anonymize when submitting
# authors:
#   - name: Anonymous

authors:
  - name: Sunbochen Tang
    affiliations:
      name: MIT AeroAstro

# must be the exact same name as your blogpost
bibliography: 2023-11-10-Physics-informed-learning-chaotic-dynamics.bib

# Add a table of contents to your post.
#   - make sure that TOC names match the actual section names
#     for hyperlinks within the post to work correctly.
toc:
  - name: Project Overivew
  - name: Problem Formulation
    subsections:
    - name: A motivating example
    - name: Physics-informed modeling
  - name: Project Scope
  - name: Research Project Disclaimers

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

## Project Overview

In this project, we would like to explore how to incorporate physics-based prior knowledge in to machine learning models for dynamical systems. Traditionally, physics laws have been used to model system behaviors with a set of differential equations, e.g. using Newton's second law to derive a pendulum equation of motion, or using Navior Stoke's equations to describe air flow in space. However, such physics-based modeling methods become challenging to implement for complex systems as all physics-based models come with assumption that helps simplify the scenario to certain extent. In recent years, machine learning has shown great potentials for developing data-driven modeling methods <d-cite key="brunton2022data"></d-cite>. 

Although learning-based methods have shown their capability of generating accuracy prediction of dynamical systems, the learned representations are difficult to interpret, especially when general multi-layer perceptron (MLP) or recurrent neural network (RNN) are used to construct the estimator. Apart from an accurate prediction, interpretability is also desirable as it helps us understand the limitation of such models. Furthermore, if a model can be constructed in line with physical modeling principles, ideally it might reveal more structured information about the data we collected from a given system. One might even hope an interpretable machine learning model would give us new insights about how to construct efficient models and discover new physics properties about a dynamical system from data.

To narrow the scope of the problem for feasibility of this course project, we will focus on the long-term prediction problem for a deterministic chaotic system, Lorenz 63, first proposed and studied in E. N. Lorenz's seminal paper <d-cite key="lorenz1963deterministic"></d-cite>. This system can be described in closed-form as a set of ordinary differential equations (ODE) with three variables, which makes the learning problem less data hungry and easier to train neural networks for its prediction with limited computation power. Despite the chaotic nature of the system (meaning that a small perturbation to the system can lead to exponential divergence in time from its original trajectory), the state of Lorenz 63 stays on a "strange attractor"" (a bounded set in the state space as shown in the animation below). We refer to the fact that the trajectory stays on the attractor as the "long-term" stability of Lorenz 63. Such long-term stability is desirable for any predictor as it indicates learning about statistical behavior of the system. Methods that can guarantee such long-term stability based on machine learning have not appeared so far, but theoretical guarantees are highly desirable as they are part of the intrinsic system properties and indicate meaningfulness of our learnt representations. Furthermore, Lorenz 63 is a simplified version of complex atmosphere thermodynamics models which are crucial in climate studies or weather forecasting. Starting with Lorenz 63 is a meaningful gateway to studying physics-informed learning approaches for climate models.

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/2023-11-10-Physics-informed-learning-chaotic-dynamics/Lorenz63.png" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    By simulating the closed-form ODE of Lorenz 63, it is observed that the trajectory always stay in the bounded region formed by these white-colored "orbits", the region is also known as the "strange attractor" of Lorenz 63. (The butterfly shape is beautiful!)
</div>

Focused on the specific Lorenz 63 system, the objective of this project is to explore machine learning model structures that attempt to achieve two goals: (1) High prediction accuracy of the state trajectory (2) Provide theoretical guarantees for long-term stability, i.e., predicted trajectory stays on the "strange attractor". In the literature, there has been approaches that use certain empirical methods to encourage long-term stability such as using noise regularization <d-cite key="wikner2022stabilizing"></d-cite>. However, such methods do not offer any theoretical guarantees and are generally difficult to interpret. We aim to investigate a specific model construction that incorporates "energy" information of the system, analogous to a recent approach in stability-guaranteed learning-based approach in control theory <d-cite key="min2023data"></d-cite>. On a high level, the proposed approach tries to learn both a predictor for Lorenz 63 and a "energy" function, and constructs a neural network for the predictor with specific activation functions such that it is constrained to a non-increasing energy condition (we will provide a more detailed description in the next section). The goal is to investigate whether this idea works on Lorenz 63 system, what type of structure we need to impose on the neural network to achieve both goals, and whether constraining the network structure leads to a trade-off between the theoretical guarantees and prediction accuracy.

## Problem Formulation

Consider a general continnuous-time nonlinear dynamics system (we will use continuous-time dynamical system formulation throughout the project):

$$ \dot{s}(t) = f(s(t)), s(t) \in \mathbb{R}^n$$

The objective of a general prediction problem is to learn a neural network-based function approximator $$g: \mathbb{R}^n \to \mathbb{R}^n$$ such that the ODE $$\dot{s}(t) = g(s(t))$$ approximates the true system above well. Namely, suppose we simulate both ODEs from the same initial condition $$r(0) = s(0)$$, we want the predicted trajectory $$r(t)$$, which is generated by $$\dot{r}(t) = g(r(t))$$ to approximate $$x(t)$$ well, i.e., $$\sup_{t \geq 0} \|r(t) - s(t)\|$$ to be small.

Specifically, here we consider the Lorenz 63 system, which can be described as (here $$x, y, z$$ are scalar variables)

$$
\begin{align*}
    \dot{x} &= \sigma (y-x)\\
    \dot{y} &= x(\rho - z) - y\\
    \dot{z} &= xy - \beta z
\end{align*}
$$

where $$\sigma, \rho, \beta$$ are scalar parameters for the system. We choose $$\sigma=10, \beta=8/3, \rho=28$$ as they generate chaotic behaviors and still observe long-term stability.

### A motivating example
We first consider a set of motivating numerical experiments which build a simple 3-layer MLP as a predictor for discrete-time Lorenz 63 to assess how difficult it is to approximate the dynamics. (Apologies for potential confusions, we use discrete-time systems because it's easier to set up, but we will use continuous-time systems in the project.) The discrete-time system is numerically integrated from the continuous-time version using 4th order Runge-Kutta method (RK4) sampled at a fixed time step $$\Delta t$$, which is in the form of

$$s[k+1] = f_d(s[k]), s[k] = s(k\Delta t) \in \mathbb{R}^3$$

We generate a dataset by sampling $$N$$ one-step pair $$(s[k], s[k+1]), k = 0, 1, 2, ..., N-1$$ from a single long trajectory using the discrete-time dynamics. A 3-layer MLP $$g(s[k]; \theta)$$ (parameterized by weights $$\theta$$) is trained to minimize the MSE loss via SGD, i.e., 

$$ \min_{\theta} \frac{1}{N} \sum_{k=0}^{N-1} \|s[k+1] - g(s[k]; \theta)\|_2^2 $$

During testing, we choose a initial condition $$s[0]$$, different than the one used to generate the training data, and generate a ground-truth trajectory of step $$N$$ as the testing dataset $$\{s[n]\}_{n=0}^{N-1}$$ and use the trained network by generating two separate trajectories as follows:

1. "MLP One-step": we apply the network to the ground-truth $$s[n]$$ at every step, i.e., the trajectory $$s_1[n]$$ that we generate satisfies $$s_1[0] = s[0]$$ and $$s_1[k+1] = g(s[k])$$.

2. "MLP Feedback": we set the initial condition $$s_2[0] = s[0]$$ and apply the network prediction iteratively, i.e., $$s_2[k+1] = g(g(... g(s[0])))$$ where $$g$$ is applied $$k$$ times.

To reduce the length of this post, we only present two most relevant examples here. When we have a dataset of $$N=1000$$ sampled one-step pairs, using GeLU activation, we are able to achieve very good prediction accuracy in both cases and both trajectories observe the "strange attractor" long-term stability as desired.
<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/2023-11-10-Physics-informed-learning-chaotic-dynamics/Gelu_1000.png" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Both prediction accuracy and long-term stability achieved when $$N=1000$$.
</div>

However, when we reduce the dataset to $$N=100$$ sampled one-step pairs, using the same GeLU activation, the "MLP feedback" trajectory fails to make accurate prediction and long-term stability. Meanwhile, the "MLP one-step" trajectory still makes very good one-step prediction. This implies that the training problem is solved almost perfectly, however, due to the nature of chaotic dynamics, a little divergence from the true dynamics, when rolled out in $$N$$ steps (as in the setting of "feedback"), it diverge from the true trajectory very quickly.
<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/2023-11-10-Physics-informed-learning-chaotic-dynamics/Gelu_100.png" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    When $$N=100$$, "MLP feedback" fails while the training problem is solved well. (the blue and green trajectories overlap with each other)
</div>

Although there are more advanced ways of learning a time-series data like this, e.g., RNNs, this simplified exercise illustrates the difficulty of learning an underlying structure of dynamics (rolling out the trajectory iteratively) compared to fitting data to achieve near-zero MSE loss (one-step prediction), especially when data is limited.

The setup of "feedback" is meaningful in a practical sense. For applications such as climate modeling, we typically want to learn what would happen in the future (in months/years), therefore, we cannot use a "one-step" prediction setup where we are restricted to predicting events in a very small future time window.

### Physics-informed modeling
As mentioned in the previous section, we aim to explore physics-informed network structures that impose certain physical constraints, with a focus on developing a method similar to the one proposed in <d-cite key="min2023data"></d-cite>. Here in the proposal, we will give a quick outline of what this approach might look like (The actual approach will be developed fully in this project).

If we look back on the Lorenz 63 equation (continuous-time), it is not difficult to see that on the right hand side, we have a second-order polynomial of the state. Therefore, if we consider the following energy function $$V$$ and write out its time derivative $$\dot{V} = dV/dt = \partial V/\partial [x, y, z] [\dot(x), \dot(y), \dot(z)]^T$$, we have

$$
\begin{align*}
    V &= \rho x^2 + \sigma y^2 + \sigma(z - 2\rho)^2\\
    \dot{V} &= -2\sigma( \rho x^2 + y^2 + \beta(z-\rho)^2 - \beta \rho^2)
\end{align*}
$$

Note that $$V$$ is always a non-negative function, and outside an ellipsoid $$E = \{(x, y, z): \rho x^2 + y^2 + \beta (z - \rho)^2 \leq \beta \rho^2\}$$, $$\dot{V}$$ is always smaller than 0, i.e., $$\forall (x, y, z) \not\in E$$, $$\dot{V}(x, y, z) < 0$$.

This is actually one interpretation why the Lorenz 63 system always stay on a bounded "strange attractor", because its trajectory always loses energy when it is outside the set $$E$$. Conceptually, the trajectory will always return to a certain energy level after it exits $$E$$.

Suppose we can construct a neural network $$g$$ for the continuous-time dynamics and another neural network $$h$$ for the energy function $$V(x, y, z)$$, i.e., 

$$
(\hat{\dot{x}}, \hat{\dot{y}}, \hat{\dot{z}}) = g(
  x, y, z
; \theta_g), \quad \hat{V}(x, y, z) = h(
  x, y, z
; \theta_h)
$$ 

In a very similar context, <d-cite key="min2023data"></d-cite> developes a specific neural network structure for $$h$$ that can ensure 

$$\dot{h} = (\partial h(x, y, z; \theta)/\partial (x, y, z)) \cdot g(x, y, z; \theta_g) < -\alpha h(x, y, z; \theta)$$

where $$\alpha > 0$$ is a positive scalar (for interested readers, this condition defines a Lyapunov function in control theory).

In this project, we aim to develop a similar structure to ensure a slightly different (local) condition:

$$\forall (x, y, z) \not\in E$$, $$\dot{\hat{V}}(x, y, z) < 0$$.

which constaints the learned model to satisfy a physical property of the system by construction. With such constraints implemented by construction, we can use the MSE loss similar to the motivating example to train both $$g$$ and $$h$$ simultaneously. Hopefully this would lead us to learning a network that achieves high prediction accuracy while obeying physical constraints.

## Project Scope
In the previous section, we gave an outline about why we want to investigate physics-based modeling for Lorenz 63 and what specific physical system information we would like to incorporate. Although we plan to spend a decent amount of time to implement and test the specific method mentioned previously, we would like to reiterate the project objective and its possible relevance to this course in this section.

The project's general objective is to investigate how to learn meaningful physics-informed representations and build constrained machine learning models that ensure certain physical properties. Picking the specific problem and approach helps us focus on a more concrete problem, but it does not restrict the project to implementation of this specific method.

More importantly, since the proposed method uses specific activation functions in <d-cite key="min2023data"></d-cite> to impose physical constraints, it restricts our model to a smaller class defined by such constraints. There could be several interesting questions downstream to be investigated:
* Would the constrained class of models be able to achieve high prediction accuracy?
* Is there a trade-off between physics constraint satisfaction (model class) and prediction accuracy (minimizing MSE loss)?
* Does the physics-informed model provide acceptable prediction accuracy in the limited data regime?
* After training, what does the $$h$$ network learn? Does it resemble an energy function?

Furthermore, we would also perform a short literature review to survey other physics-informed learning methods for dynamical systems. If we find a highly relevant approach that would work for problem, under the time constraint of the project, we will try to implement such approaches and compare our approach with them as well.

## Research Project Disclaimers
I would like to inform the teaching staff that this project is planned to be part of my ongoing research. During the semester, I don't have much time to work on this idea as I am trying to meet a conference deadline for another ongoing project. Since the project explores learning efficient physical representations for dynamical system, I am hoping that I can use the course project opportunity to work on this idea. There has not been much prior work done except the thought process presented in this proposal. If the specific approach proposed turns out to be successful, I would like to extend it into my next research project and hopefully part of my Ph.D. thesis.

Please let me know if this would be acceptable under the course guideline. I'd be happy to make other modifications to follow the course project guideline on using ideas relevant to ongoing/future research.