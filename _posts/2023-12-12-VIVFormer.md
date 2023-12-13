---
layout: distill
title: VIVFormer
description: A deep transformer framework for forecasting extended horizons of high-frequency non-stationary time-series. Applications and insights drawn from vortex induced vibrations data collected at the MIT Towing Tank.

date: 2022-12-01
htmlwidgets: true

# Anonymize when submitting
# authors:
#   - name: Anonymous

authors:
  - name: Andreas Mentzelopoulos
    url: "https://scholar.google.com/citations?user=0SOhn-YAAAAJ&hl=en"
    affiliations:
      name: PhD Candidate in Mechanical Engineering and Computation, MIT


# must be the exact same name as your blogpost
bibliography: 2023-12-12-VIVFormer.bib  

# Add a table of contents to your post.
#   - make sure that TOC names match the actual section names
#     for hyperlinks within the post to work correctly.
toc:
  - name: Proposal


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

## Proposal

Vortex induced vibrations (VIV) are vibrations that affect bluff bodies in the presence of currents. VIV are driven by the periodic formation and shedding of vortices in the bodies' wakes which create an alternating pressure variation causing persistent vibrations  <d-cite key="triantafyllou2016vortex"></d-cite>. The vibration amplitude in VIV is typically moderate, not exceeding about one body diameter <d-cite key="bernitsas2019eigen"></d-cite>. For flexible bodies, VIV are not uniform along the body's length (usally refered to as the span) but rather different points along span vibrate with different amplitudes and phases. 

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/2023-12-12-VIVFormer/Intro.jpg" class="img-fluid" %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/2023-12-12-VIVFormer/Intro2.jpeg" class="img-fluid" %}
    </div>
</div>
<div class="caption">
    Schematic diagrams of vortex induced vibrations of flexible bodies.
</div>

VIV have become a problem of interest to both theoreticians, due to the complex underlying mechanisms involved <d-cite key="WILLIAMSON1988355"></d-cite>, and engineers, due to the practical significance of mitigating the fatigue damage VIV can cause to offshore structures and equipment such as marine risers and offshore wind turbines <d-cite key="vandiver2006fatigue"></d-cite>. 

Semi-empirical models are the offshore industry's standard approach to VIV modelling. Specifically, semi-empirical models <d-cite key="zheng2011vortex, vandiver1999shear7, larsen2001vivana"></d-cite> whose foundations are physics based have been successful in predicting flexible body VIV on average (i.e. estimating the average of the vibration as a function of body location for many cycles). However, such models' accuracy relys heavily on the empirical coefficients used and obtaining such coefficients requires many (expensive) experiments in towing tanks or wind tunnels. In addition, the models cannot continuously predict VIV motions but rather can only inform about averages.

Forecasting the time-series of VIV of flexible bodies has only recently been attempted. Instead of using physics based methods, Kharazmi et al.(2021) used a data-driven approach and predicted a few cycles of the vibration in the future with reasonable accuracy using LSTM networks in modal space (LSTM-ModNet) <d-cite key="kharazmi2021data"></d-cite>. Albeit a powerful framework, the LSTM-Modnet can handle a single location along the body, and as such, predicting more than one locations requires extensive amounts of computational resources for training multiple LSTM-ModNets (one for each location of interest). 

Although leveraging transformers to expand the horizon of predictions of time series is a very active field of research <d-cite key="zhou2021informer, zeng2023transformers, liu2022non, zhou2022fedformer"></d-cite>, transformers have not yet been used to predict VIV of flexible bodies, which are real high-frequency non-stationary time-series, to the best of the author's knowledge. In this work, an attempt will be made to develop a transformer architecture to predict the VIV motions of a flexible body using data collected at the MIT Towing tank. 

In the scope of this work, the effects of single versus muti-headed attention, attention dimension, and number of MLP layers used in the architecture will be examined. In addition the effect of masking attention in order constraint (or rather more effectively guide) information flow within the architecture is of particular interest. Additional questions of interest would be to explore whether embeddings could be added or learned to enhance the transformer's performance.




