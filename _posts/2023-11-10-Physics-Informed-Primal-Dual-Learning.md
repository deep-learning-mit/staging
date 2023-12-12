---
layout: distill
title: Physics-Informed Primal-Dual Learning
description: Learning a deep net to optimize an LP, subject to both primal and dual hard constraints. Exploration of a novel proposed KKT-based training scheme.
date: 2023-11-08
htmlwidgets: true

# Anonymize when submitting
# authors:
#   - name: Anonymous

authors:
  - name: Thomas Lee
    url: "https://www.linkedin.com/in/thomas-lee-2017/"
    affiliations:
      name: MIT

# must be the exact same name as your blogpost
bibliography: 2023-11-10-Physics-Informed-Primal-Dual-Learning.bib  

# Add a table of contents to your post.
#   - make sure that TOC names match the actual section names
#     for hyperlinks within the post to work correctly.
toc:
  - name: Proposal

---

<b>Motivation</b>: Physics-informed machine learning has emerged as an important paradigm for safety-critical applications where certain constraints must be satisfied. One such application domain is energy systems. But an additional feature of energy markets is that prices are also a crucial feature that affects system efficiency and investment incentives. How can both physical operations (primal) and pricing (dual) constraints be satisfied?

The goal of this project is to learn a deep learning surrogate for a linear programming optimization problem with hard constraints. The overall approach is inspired by standard KKT conditions, and specifically the interior point approach of incrementally tighteting the relaxed complementarity condition <d-cite key="gondzio2012interior"></d-cite>.

Training will be done in a self-supervised manner, where input vectors $$x = (c,A,b)$$ (i.e. parameters in the LP) are provided. The proposed method will predict output vectors consisting of both primal and dual solutions: $$(y,\lambda)$$. During training, the method will maintain both primal and dual feasibility through a combination of equality completion <d-cite key="donti2021dc3"></d-cite> and the recent application of gauge maps (i.e. based on the Minkowski function) <d-cite key="zhang2023efficient"></d-cite>, both of which have been successfully applied to deep learning. Finally, the only remaining KKT condition is complementary slackness, which I propose to drive towards 0 using a custom differentiable "bilinear loss" layer (in a self-supervised manner):

$$\mathcal{L}(x,y,\lambda) = \sum_k (A_k y_k - b_k)^T \lambda_k$$

The main conceptual novelty here is to combine both primal constraints (a la physics-informed or safe ML), as well as dual feasibility - which intuitively could help to push towards an inductive bias for optimality. (While a supervised or self-supervised approach may use the primal objective as the loss function, a hypothesis is the the novel dual-feasibility condition might help better "pull" the predictions towards optimality on out of sample inputs). This approach might offer advantages over previous attempts in the literature, which overall are able to obtain reasonable primal feasibility but may still suffer from suboptimality.
- DC3 <d-cite key="donti2021dc3"></d-cite> approach requires an inner feasibility gradient descent, which requires additional hyperparamter tuning beyond the deep learning parameters. Insufficient number of descent steps could still lead to primal infeasibility (e.g. Table 2 of <d-cite key="li2023learning"></d-cite>).
- DC3 has been shown to sometimes exhibit significant suboptimality on quadratic programming problems. It is unclear whether this is empirically a problem for LPs; nonetheless, there is no optimality guarantee. Instead, the proposed approach here would be able to provide valid primal-dual optimality bounds at every training step (e.g. as a stopping criterion) and testing step (e.g. to indicate regions where additional training may be needed).  
- Active set learning approach <d-cite key="pagnier2022machine"></d-cite>, i.e. predict primal active constraints, will satisfy complementary slackness by construction. The resulting duals are coherent since the solution comes from solving the completed KKT system. But may not be primal feasible if the active set prediction has false negatives.
- Older price prediction approaches <d-cite key="liu2021graph"></d-cite>, i.e. predict dual values and then infer primal solution, similarly also satisfies complementary slackness by construction. Again these are not guaranteed to be primal feasible; moreover the dual prices may not be coherent.
- Does not require an outer loop (with additional hyperparameters e.g. penalty and learning rates) as in having 2 separate networks in this primal-dual ALM-type approach <d-cite key="park2023self"></d-cite>. 
- Importantly, directly provides a set of coherent dual outputs, which can be directly important for applications (e.g. predicting electricity prices) or used in a downstream task (e.g. duals for Benders decomposition). A primal-only feasible neural net could potentially be used to calculate the gradient in a backward step; but this may require more memory than if both primal and dual values are predicted during the forward step (with no_grad).


Mathematically, the main challenge is that the proposed bilinear loss is clearly nonconvex, which might (or might not) lead to SGD optimization convergence issues. Some previous work do use bilinear loss or bilinear layers<d-cite key="shazeer2020glu"></d-cite><d-cite key="resheff2017controlling"></d-cite>, suggesting this could potentially work empirically.

In terms of coding implementation, the main tasks are to
1. Implement the gauge map differentiable layer. This open source colab notebook could be a good start: https://github.com/zhang-linnng/two-stage-dcopf-neural-solver/
2. Implement the bilinear loss differentiable layer. (The derivative field is $$(y,x)$$.)