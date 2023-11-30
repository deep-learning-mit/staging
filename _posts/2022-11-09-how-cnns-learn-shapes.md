---
layout: distill
title: How CNNs learn shapes
description: 
date: 2023-11-09
htmlwidgets: true

# Anonymize when submitting
# authors:
#   - name: Anonymous

authors:
  - name: Chloe Hong
    url: 
    affiliations:
      name: MIT

# must be the exact same name as your blogpost
bibliography: 2022-11-09-how-cnns-learn-shapes.bib  

# Add a table of contents to your post.
#   - make sure that TOC names match the actual section names
#     for hyperlinks within the post to work correctly.
toc:
  - name : Background
  # - name: Equations
  # - name: Images and Figures
  #   subsections:
  #   - name: Interactive Figures
  # - name: Citations
  # - name: Footnotes
  # - name: Code Blocks
  # - name: Layouts
  # - name: Other Typography?

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

## Background

One widely accepted intuition is that CNNs combines low-level features (e.g. edges) to gradually learn more complex and abstracted shapes to detect objects while being invariant to positional and translation.

> As [@kriegeskorte2015deep] puts it, “the network acquires complex knowledge
about the kinds of shapes associated with each category. [...] High-level units appear to learn
representations of shapes occurring in natural images” (p. 429). This notion also appears in other
explanations, such as in [@lecun2015deep]: Intermediate CNN layers recognise “parts of familiar
objects, and subsequent layers [...] detect objects as combinations of these parts” (p. 436). We term
this explanation the shape hypothesis.
As a result, the final prediction is based on global patterns rather than local features.

However, there has been contradictory findings that CNNs trained on off-the-shelf datasets are biased towards predicting the category corresponding to the texture rather than shape. [@geirhos2018imagenet]

{% raw %}{% include figure.html path="assets/img/2023-11-09-how-cnns-learn-shapes/shapetexture.png" class="img-fluid" %}{% endraw %}

Going further, previous works have suggested ways to increase the shape bias of CNNs including data augmentation and relabelling. 
While these works have successfully shown the discriminative bias of CNNs toward certain features, they do not identify how the networks "perception" changes. 
With this project, I seek to evaluate the bias contained (i) in the latent representations, and (ii) on a per-pixel level. 



## Methods
I choose two approaches from [@geirhos2018imagenet] and [@chung2022shape] that augment the dataset to achieve an increased shape bias in CNNs. 
To gain a better understanding what type of shape information contained in the network is discriminative, where shape information is encoded, as well as when the network learns about object shape during training, I use an optimization method to visualize features learned at each layer of the trained models. 
By comparing the original model to the augmented version, and across different augmentation methods, we can evaluate if there is a common pattern in the way CNNs learns shapes and what additional information is most effective in increasing shape bias in CNNs.

### Data augmentations
[@geirhos2018imagenet] increased shape bias by augmenting the data with shape-based representations. 

| Features      | Dataset                               |
|---------------|---------------------------------------|
| image         | ImageNet                              |
| image + shape | ImageNet augmented with line drawings |
| shape         | Line drawings                         |

 [@chung2022shape] speculates data distribution is the root cause of discriminative biases in CNNs. To address this, they suggested a granular labeling scheme that redesigns the label space to pursue a balance between texture and shape biases. 

| Labels        | Dataset                               |
|---------------|---------------------------------------|
| categorical   | ImageNet                              |
| categorical + style | ImageNet                        |


### CNN feature visualization 
We visualize features that are understood by the CNN model at the layer level using the following optimization framework.

{% raw %}{% include figure.html path="assets/img/2023-11-09-how-cnns-learn-shapes/cnnfeaturevisualization.png" class="img-fluid" %}{% endraw %}

