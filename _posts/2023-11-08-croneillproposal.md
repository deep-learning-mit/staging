---
layout: distill
title: How to learn a linear representation of a dynamical system
description: A final project proposal for 6.s898 in fall 2023
date: 2023-11-08
htmlwidgets: true

authors:
  - name: Cormac O'Neill
    url: 
    affiliations:
      name: MIT, Cambridge

# must be the exact same name as your blogpost
bibliography: 2023-11-08-croneillproposal.bib  

# Add a table of contents to your post.
#   - make sure that TOC names match the actual section names
#     for hyperlinks within the post to work correctly.
toc:
  - name: Proposal
---


## Proposal

Linear system representations offer numerous benefits for analysis and control. Unfortunately, we live in a world where most interesting dynamic systems are inherently nonlinear. Traditionally engineers have linearized nonlinear systems by truncating a Taylor series approximation of the dynamics about a point. While this technique can be useful, it is an inherently point-wise approach. In contrast, recent work has investigated how lifting linearization techniques can be used as an alternative. Underpinned by Koopman operator theory, lifting linearization expands a nonlinear system to a higher dimension by appending nonlinear functions of its state to the system’s representation <d-cite key="brunton2021modern"></d-cite>. One of the primary open questions in the field is how to best select these nonlinear functions (referred to as “observable functions”). A recent, popular approach is to learn the observable functions from data with a neural network <d-cite key="lusch2018deep, yeung2019learning, abraham2019active, han2020deep"></d-cite>. This network usually takes on the form of an autoencoder with a representation space that is a higher dimension than the input.

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/2023-11-08-croneillproposal/deepnet.png" class="img-fluid" %}
    </div>
</div>
<div class="caption">
    An example of a neural network architectured used to learn observables for a linear Koopman model, taken from <d-cite key="lusch2018deep"></d-cite>
</div>

For this project, I want to investigate how deep learning can be used to learn more effective observable functions. I am especially interested in studying how to learn observables for piecewise dynamical systems:

* Can a curriculum learning-inspired approach lead to observables with varying spatial frequencies? Can we first train an autoencoder to learn a small number of observables that are better at representing the system’s averaged global dynamics at the expense of local accuracy? If we then add additional width to the network and continue training, will we be able to learn observables that are more focused on particular regions of state space?

* If observables are separately trained on different regions of state space, can they then be concatenated to provide a better dynamic model? This approach is inspired by work from a previous lab mate of mine <d-cite key="ng2022learned"></d-cite>.

I plan to take an ablative approach to studying these questions by training three different models: a standard network for learning observables that works on the full training data set, the above curriculum learning approach, and then finally an approach that uses observables trained separately on different regions of state space. I will then compare the performance of the resulting observables in predicting the trajectory of a dynamical system.

I am also considering some additional questions that could be interesting, although they are less well thought out:

* How can the autoencoder structure of observable generators be modified to improve performance? I need to do further literature review, but I do not believe that there has been a quantitative analysis of how network architecture (such as the type of activation function, the importance of depth) impacts performance. I am not even sure if skip connections have been utilized in prior work.

* Are there alternatives to fully-connected layers that could be useful for generating observable functions? I have given this question much less thought, but it is a topic I would love to discuss with the TAs. Certain lifted linearization approaches (dynamic mode decomposition) work by taking measurements throughout the state space and using them as observables. For example, a highly nonlinear fluid flow can be linearized by taking measurements throughout the fluid. This creates a data structure that reminds me of images, causing me to wonder if a convolutional or transformer inspired approach could have some use in this field.
