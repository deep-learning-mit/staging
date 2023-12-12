---
layout: distill
title: Prompt to Prompt
description: Text-based image editing via cross-attention mechanisms - the research of hyperparameters and novel mechanisms to enhance existing frameworks
date: 2023-11-07
htmlwidgets: true

# Anonymize when submitting
# authors:
#   - name: Anonymous

authors:
  - name: Carla Lorente
    url: "https://www.linkedin.com/in/carla-lorente/"
    affiliations:
      name: MIT EECS 2025
  - name: Linn Bieske
    url: "https://www.linkedin.com/in/linn-bieske-189b9b138//"
    affiliations:
      name: MIT EECS 2025

# must be the exact same name as your blogpost
bibliography: 2023-11-07-prompt-to-prompt.bib   #############CHANGED!!!!!!!!!!!!!!

# Add a table of contents to your post.
#   - make sure that TOC names match the actual section names
#     for hyperlinks within the post to work correctly.
toc:
  - name: Introduction
  - name: Research questions
  - name: Methodology
  - name: Conclusion
  - name: References

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

## Introduction

Recently, the techniques to edit images have advanced from methodologies that require the user to edit individual pixels to deep learning-based image editing. The latter employ for example large image generation models (e.g., stable diffusion models). While these deep learning-based image editing techniques initially required the user to mark particular areas which should be edited  (Nichol et al., 2021 <d-cite key="nichol2021glide"></d-cite>; Avrahami et al., 2022a<d-cite key="avrahami2022blendeddiffusion"></d-cite>; Ramesh et al., 2022), recently the work by (Hertz et al, 2022 <d-cite key="hertz2022prompttoprompt"></d-cite>) has shown that this becomes unnecessary. Instead, image editing can be performed using a cross-attention mechanism. In particular, the proposed prompt-to-prompt editing framework enables the controlling of image edits by text only. The section below provides an overview of how this prompt-to-prompt framework works (Figure 1, by (Hertz et al, 2022<d-cite key="hertz2022prompttoprompt"></d-cite>)). 

{% include figure.html path="assets/img/2023-11-07-prompt-to-prompt/1-cross_attention_masks.png" class="img-fluid" %}

*Figure 1: Cross-attention method overview. Top: visual and textual embedding are fused using cross-attention layers that produce attention maps for each textual token. Bottom: we control the spatial layout and geometry of the generated image using the attention maps of a source image. This enables various editing tasks through editing the textual prompt only. When swapping a word in the prompt, we inject the source image maps Mt, overriding the target maps M ∗ t . In the case of adding a refinement phrase, we inject only the maps that correspond to the unchanged part of the prompt. To amplify or attenuate the semantic effect of a word, we re-weight the corresponding attention map. (Hertz et al, 2022 <d-cite key="hertz2022prompttoprompt"></d-cite>).* 

While this proposed framework has significantly advanced the image editing research field, its performance leaves still room for improvement such that open research questions remain. For example, when performing an image editing operation that changes the hair color of a woman, significant variability across the woman’s face can be observed (Figure 2). This is undesirable, as the user would expect to see the same female face across all four images. 

{% include figure.html path="assets/img/2023-11-07-prompt-to-prompt/2-Experimentation_proposed_prompt_to_prompt.png" class="img-fluid" %}

*Figure 2: Experimentation with the proposed prompt-to-prompt image editing framework presented by (Hertz et al, 2022<d-cite key="hertz2022prompttoprompt"></d-cite>). The faces of the women show significant variability even though they should remain invariant across all four generated/ edited images.*

Within our work, we will start to further benchmark the performance of the proposed framework, explore the impact of its hyperparameters on the image editing process, and research opportunities to improve the underlying cross-attention mechanism. 


## Research questions

Our research question is threefold and contains both realistic and ambitious aspects.


<ul>
  <li><strong>Benchmark:</strong> First, we intend to further benchmark the capabilities of the proposed framework (e.g., across defined dimensions such as applicability to different domains, robustness of editing, realism, and alignment to user prompt and intention).</li>
  <li><strong>Hyperparameter investigation:</strong> Second, the currently proposed prompt-to-prompt framework does not explore and quantify the impact of its different hyperparameters on its editing performance (time steps of diffusion for each cross-attention mask, scaling factor, …)</li>
  <li><strong>Enhanced attention mechanism:</strong> Initial evaluation of the prompt-to-prompt framework made us observe shortcomings including the distortion of the image across editing steps. Therefore, we will explore approaches to strengthen the underlying cross-attention mechanism (e.g., by exploring regularization techniques). The exact mechanism which could lead to an enhanced image editing performance is subject to research.</li>
</ul>



## Methodology

To perform our research, we plan to build upon the code which complemented the paper published by (Hertz et al, 2022 <d-cite key="hertz2022prompttoprompt"></d-cite>, [Link to code]( https://github.com/google/prompt-to-prompt/)). Concretely, we will rely on a stable diffusion model from hugging face which we will access via Python. No model training is required as we will solely work with attention layers that capture spatial information about the images.  By now, we have reviewed and tested the code implementation, resolved any encountered bugs, and have started the exploration of the functionalities of the published repository. This makes us feel comfortable that our ambitions are feasible. 


To achieve all three of our realistic and ambitious research goals we plan to undertake the following steps: 
<ul>
  <li><strong>Benchmarking:</strong> First, we will define 5 categories of interests (e.g., human faces, interior designs, animals, food, and transportation) for which we will test both, the image generation process of the stable diffusion model itself as well as the image editing performance of the cross-attention mechanisms presented by (Hertz et al, 2022 <d-cite key="hertz2022prompttoprompt"></d-cite>). The judge of the benchmarking process will be ourselves (Carla and Linn), since this will help us further understand the shortcomings of the existing framework.</li>
  <li><strong>Hyperparameter investigation:</strong> For a selection of the defined categories of interest we will perform a hyperparameter study. This will entail two scenarios: 1. studying the impact of each individual hyperparameter independently to research its individual impact on the quality of the edited images. 2. Studying the interdependence of the hyperparameters by performing a grid search. The outcome of step (1) would inform reasonable search spaces for each hyperparameter.</li>
  <li><strong>Enhanced attention mechanism:</strong> We have the ambition to explore opportunities to improve the performance of the cross-attention image editing mechanism beyond the tuning of hyperparameters. Therefore, we will research approaches to improve the framework. Each architecture change of the cross-attention algorithm will be benchmarked to assess whether a performance improvement is possible. Here, we may look into expanding the user input to a larger group of people beyond our team</li>
</ul>



## Conclusion
This research endeavors to push the boundaries of text-based image editing, with the potential to significantly streamline creative workflows and introduce a new level of user accessibility to image manipulation. By delving into the intricacies of the prompt-to-prompt framework and its underlying hyperparameters, the research not only paves the way for more robust and realistic image manipulations but also opens up new avenues for creative expression and accessibility in digital media.

