---
layout: distill
title: CNN Activation Patching Proposal
# description: 
date: 2023-11-10
htmlwidgets: true

# Anonymize when submitting
# authors:
#   - name: Anonymous

authors:
  - name: Shariqah Hossain
    # url: "https://en.wikipedia.org/wiki/Albert_Einstein"
    affiliations:
      name: MIT

# must be the exact same name as your blogpost
bibliography: 2023-11-10-CNN-activation-patching.bib  

# Add a table of contents to your post.
#   - make sure that TOC names match the actual section names
#     for hyperlinks within the post to work correctly.
# toc:
#   - name: Citations

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

Neural nets contain large amounts of parameters and connections that they use to model a given phenomena. Often, the breadth and complexity of these systems make it difficult for humans to understand the mechanisms that the model uses to perform its tasks. The model is treated like a black-box, often leading to a reliance on trial-and-error and larger, more diverse datasets to alter the behavior of the model when it does not behave in the desired way.

Mechanistic interpretability aims to unpack the underlying logic and behaviors of neural networks. <d-cite key="zhang2023best"></d-cite> Activation patching is an interpretability technique that replaces activations in a model with that of another model in order to analyze its role in the output. When a patched activation changes the output of the first model to match that of the second, it indicates that the patched activation is responsible for that task within the model. <d-cite key="Vig2020InvestigatingGB"></d-cite> For example, if the attention head of a transformer model with prompt "The Eiffel Tower is in" is patched into a model with "The Colosseum is in" and successfully changes the output from "Rome" to "Paris", this indicates that the patched head contains information about the Eiffel Tower. <d-cite key="meng2023locating"></d-cite>

{% include figure.html path="assets/img/2023-11-10-CNN-activation-patching/patch.png" class="img-fluid" %}<d-cite key="meng2023locating"></d-cite>

Activation patching has been used for language models, but this project proposes to apply this technique to a vision transformer. <d-cite key="dosovitskiy2021image"></d-cite> Similar to the strategies of existing language model work, there can be patching with a clean and corrupted dataset. For example, a clean model can be trained on color images and a corrupted one can be trained only with black and white images. If an attention head from the color model is patched into the black and white model such that it can correctly classify images, that will reveal that the head contains information about the color of the image. Similar analysis can be performed for other tasks vision models perform before classification, such as edge detection. In this case, the corrupted model can have blurred images.

Given successful analysis of simpler features of the images, it would be interesting to explore how attribution patching can reveal information about vision models with racial and gender bias. Datasets that focus on emotion and actions may have biases when determining people of which demographic are most likely to make a given expression or perform a given action. The mechanisms behind this bias can be dissected further through activation patching. <d-cite key="Vig2020InvestigatingGB"></d-cite>

CIFAR-100 is a potential dataset that can be used for this study. The images can be blurred or changed to black and white as needed. 

Thank you to Saachi Jain for helping me flesh out this idea.