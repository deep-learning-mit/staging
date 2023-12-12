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
  - name: Literature Review
  - name: Outline of our research
  - name: A. Hyperparameter Study of prompt-to-prompt editing method "word swap"
  - name: A1. Exploration of silhouette threshold hyperparameter ("k")
  - name: A2. Exploration of cross-attention injection hyperparameter ("cross replace steps")
  - name: A3. Exploration of self-attention hyperparameter ("self replace steps")
  - name: A4. Cycle Consistency of method
  - name: B. Generalization of optimized hyperparameters to "attention re-weight method"
  - name: Our proposed method
  - name: Future work
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

Recently, the techniques to edit images have advanced from methodologies that require the user to edit individual pixels to deep learning-based image editing. The latter employ for example large image generation models (e.g., stable diffusion models). While these deep learning-based image editing techniques initially required the user to mark particular areas that should be edited  (Nichol et al., 2021 <d-cite key="nichol2021glide"></d-cite>; Avrahami et al., 2022a<d-cite key="avrahami2022blendeddiffusion"></d-cite>; Ramesh et al., 2022), recently the work by (Hertz et al, 2022 <d-cite key="hertz2022prompttoprompt"></d-cite>) has shown that this becomes unnecessary. Instead, image editing can be performed using a cross-attention mechanism. In particular, the proposed prompt-to-prompt editing framework enables the controlling of image edits by text only. The section below provides an overview of how this prompt-to-prompt framework works (Figure 1, by (Hertz et al, 2022<d-cite key="hertz2022prompttoprompt"></d-cite>)). 

{% include figure.html path="assets/img/2023-11-07-prompt-to-prompt/1-cross_attention_masks.png" class="img-fluid" %}

*Figure 1: Cross-attention method overview. Top: visual and textual embedding are fused using cross-attention layers that produce attention maps for each textual token. Bottom: we control the spatial layout and geometry of the generated image using the attention maps of a source image. This enables various editing tasks through editing the textual prompt only. When swapping a word in the prompt, we inject the source image maps Mt, overriding the target maps M ∗ t . In the case of adding a refinement phrase, we inject only the maps that correspond to the unchanged part of the prompt. To amplify or attenuate the semantic effect of a word, we re-weight the corresponding attention map. (Hertz et al, 2022 <d-cite key="hertz2022prompttoprompt"></d-cite>).* 

While this proposed framework has significantly advanced the image editing research field, its performance leaves still room for improvement such that open research questions remain. For example, when performing an image editing operation that changes the hair color of a woman, significant variability across the woman’s face can be observed (Figure 2). This is undesirable, as the user would expect to see the same female face across all four images. 

{% include figure.html path="assets/img/2023-11-07-prompt-to-prompt/2-Experimentation_proposed_prompt_to_prompt.png" class="img-fluid" %}

*Figure 2: Experimentation with the proposed prompt-to-prompt image editing framework presented by (Hertz et al, 2022<d-cite key="hertz2022prompttoprompt"></d-cite>). The faces of the women show significant variability even though they should remain invariant across all four generated/ edited images.*

Within our work, we will start to further benchmark the proposed framework's performance, explore its hyperparameters' impact on the image editing process, and research opportunities to improve the current performance.

## Literature Review

Before delving into the details of the prompt-to-prompt editing method, let's briefly recap some existing techniques to edit images with diffusion models that have paved the way for this revolutionary approach:

### 1. Adding noise to an image and denoising with a prompt ###

In **SDEdit: Guided Image Synthesis and Editing with Stochastic Differential Equations** <d-cite key="meng2021sdedit"></d-cite> , the user takes an image, introduces noise and then denoises it according to a user-provided prompt. As an example, given an image, users can specify how they want the edited image to look using pixel patches copied from other reference images.

{% include figure.html path="assets/img/2023-11-07-prompt-to-prompt/literature_review/SDEdit.png" class="img-fluid" %}

A similar approach is used in the paper **MagicMix: Semantic Mixing with Diffusion Models** <d-cite key="liew2022magicmix"></d-cite> which uses a pre-trained text-to-image diffusion based generative model to extract and mix two semantics. The figure below showcases the detailed pipeline of MagicMix (image-text mixing). Given an image x<sub>0</sub> of layout semantics, they first craft its corresponding layout noises from step Kmin to K<sub>max</sub>. Starting from K<sub>max</sub>, the conditional generation process progressively mixes the two concepts by denoising given the conditioning content semantics (“coffee machine” in this example). For each step k in [K<sub>min</sub>; K<sub>max</sub>], the generated noise of mixed semantics is interpolated with the layout noise x<sub>k</sub> to preserve more layout details.

<div style="text-align:center;">
  {% include figure.html path="assets/img/2023-11-07-prompt-to-prompt/literature_review/corgi_coffee_machine_1.png" class="img-fluid" width="100" %}
</div>

{% include figure.html path="assets/img/2023-11-07-prompt-to-prompt/literature_review/corgi_coffee_machine_2.png" class="img-fluid" %}

### 2. Take an image, add noise and denoise it with a prompt + Add a mask ###

In the paper **Blended Diffusion: Text-Driven Editing of Natural Images** <d-cite key="avrahami2022blended"></d-cite> , given an input of an image and a mask, the blended diffusion modifies the masked area according to a guided text prompt, without affecting the unmasked regions. One limitation of this is that it relies on the user having to produce this mask to indicate the editing region.

{% include figure.html path="assets/img/2023-11-07-prompt-to-prompt/literature_review/Blended_Difussion.png" class="img-fluid" %}

An advanced version of this diffusion mode is discussed in the paper **Text-based inpainting with CLIPSef and Stable Diffusion** <d-cite key="luddecke2022image"></d-cite>. In this paper, the novelty is that the user doesn't have to do the mask manually. Instead, it can use an existing segmentation model (e.g. ClipSef). Another alternative is presented in the paper **DiffEdit: Diffusion-based semantic image editing with mask guidance** <d-cite key="couairon2022diffedit"></d-cite> where the mask is generated directly from the diffusion model.

### 3. Fine-tune (“overfit”) on a single image and then generate with the fine-tuned model ###

In the paper **Imagic: Text-based real image editing with diffusion models** <d-cite key="kawar2023imagic"></d-cite> and **Unitune: Text-driven image editing by fine-tuning a diffusion model on a single image** <d-cite key="valevski2023unitune"></d-cite>, the authors perform extensive fine-tuning on either the entire diffusion model or specific sections of it. This process is computationally and memory-intensive, setting it apart from alternative methods.

{% include figure.html path="assets/img/2023-11-07-prompt-to-prompt/literature_review/Fine_Tuning.png" class="img-fluid" %}

### Prompt-to-prompt
The prompt-to-prompt editing method is a significant advancement compared with the existing image editing techniques that rely on diffusion models. Unlike the methods explained above that involve adding noise, using masks, or fine-tuning, the prompt-to-prompt method stands out because of its simplicity, flexibility, and user-friendliness. In the former methods, users often face challenges such as manually creating masks or undergoing resource-intensive fine-tuning processes, which can be both time-consuming and technically demanding. In contrast, the prompt-to-prompt editing method streamlines the editing process by allowing users to directly specify their desired edits through language prompts. This approach eliminates the need for intricate masking or extensive model training as well as leverages the power of human language to precisely convey editing intentions. 

Throughout our research, we will adopt the prompt-to-prompt editing method as our starting point, with the aim of enhancing its performance.

## Outline of our research

To perform our research, we plan to build upon the code which complemented the paper published by (Hertz et al, 2022 <d-cite key="hertz2022prompttoprompt"></d-cite>, [Link to code]( https://github.com/google/prompt-to-prompt/)). Concretely, we will rely on a stable diffusion model from hugging face which we will access via Python. No model training is required as we will solely work with attention layers that capture spatial information about the images.

Our study will be divided into 3 main subsections:
{% include figure.html path="assets/img/2023-11-07-prompt-to-prompt/analysis/Figure 01 - outline.png" class="img-fluid" %}


## A. Hyperparameter Study of prompt-to-prompt editing method "word swap"

In the forthcoming subsection, we delve into a comprehensive analysis of the hyperparameters pertaining to the "word swap" method within the prompt-to-prompt editing framework. Before delving into the specifics, it's crucial to understand the significance of these hyperparameters and their default values, as originally outlined in the seminal work by Hertz et al<d-cite key="hertz2022prompttoprompt"></d-cite>. 

{% include figure.html path="assets/img/2023-11-07-prompt-to-prompt/analysis/Figure 02 - outline section A.png" class="img-fluid" %}

{% include figure.html path="assets/img/2023-11-07-prompt-to-prompt/analysis/Figure 03 - Local editing.png" class="img-fluid" %}

{% include figure.html path="assets/img/2023-11-07-prompt-to-prompt/analysis/Figure 04 - Cross replace steps explanation.png" class="img-fluid" %}

{% include figure.html path="assets/img/2023-11-07-prompt-to-prompt/analysis/Figure 05 - Sel-attention explanation.png" class="img-fluid" %}

We will systematically explore various hypotheses regarding each hyperparameter and present our empirical findings, shedding light on their individual impacts on the editing process. This examination aims to provide valuable insights into optimizing the performance of the "word swap" method and enhancing its practical utility.

{% include figure.html path="assets/img/2023-11-07-prompt-to-prompt/analysis/Figure 06 - Hypothesis and findings.png" class="img-fluid" %}


## A1. Exploration of silhouette threshold hyperparameter ("k")

In this section, we embark on an exploration of the silhouette threshold hyperparameter ("k"). We aim to unravel the influence of varying this parameter while using the prompt '_"A woman's face with blond hair"_' and making alterations to different hair colors (brown, red, black). The GIF below showcases the representation of these experiments.

{% include figure.html path="assets/img/2023-11-07-prompt-to-prompt/GIFs/threshold_k/change_threshold_womens_face2.gif" class="img-fluid" %}

{% include figure.html path="assets/img/2023-11-07-prompt-to-prompt/analysis/Figure 07 - Results silhouette parameter k - faces.png" class="img-fluid" %}

Additionally, we present a comparative analysis of the impact of this hyperparameter on editing tasks related to landscapes. For instance, we employ the prompt '_"A river between mountains"_' and manipulate the landscape, including options like streets, forests, and deserts. The results of this landscape-oriented analysis can be seen in the figure below.

{% include figure.html path="assets/img/2023-11-07-prompt-to-prompt/GIFs/threshold_k/attention_replace_rivers.gif" class="img-fluid" %}

{% include figure.html path="assets/img/2023-11-07-prompt-to-prompt/analysis/Figure 08 - Results silhouette parameter k - landscape.png" class="img-fluid" %}



## A2. Exploration of cross-attention injection hyperparameter ("cross replace steps")

Below we showcase the effect of the silhouette threshold hyperparameter ("k") and the cross-attention injection hyperparameter("cross_replace_steps"). We manipulate the "k" value, setting it to 3 different levels: 0, 0.3 (default literature value), and 0.6. The experiment was performed for both women's faces and landscapes, providing a comprehensive understanding of how these hyperparameters affect the editing process. The following GIFs showcase the results of our exploration.

### With k = 0:

{% include figure.html path="assets/img/2023-11-07-prompt-to-prompt/GIFs/cross_replace_steps/k_zero_cross_replace_women.gif" class="img-fluid" %}
{% include figure.html path="assets/img/2023-11-07-prompt-to-prompt/GIFs/cross_replace_steps/k_zero_cross_replace_river.gif" class="img-fluid" %}

### With k = 0.3:
{% include figure.html path="assets/img/2023-11-07-prompt-to-prompt/GIFs/cross_replace_steps/k_point3_cross_replace_women.gif" class="img-fluid" %}
{% include figure.html path="assets/img/2023-11-07-prompt-to-prompt/GIFs/cross_replace_steps/k_point3_cross_replace_river.gif" class="img-fluid" %}

### With k = 0.6:

{% include figure.html path="assets/img/2023-11-07-prompt-to-prompt/GIFs/cross_replace_steps/k_point6_cross_replace_women.gif" class="img-fluid" %}
{% include figure.html path="assets/img/2023-11-07-prompt-to-prompt/GIFs/cross_replace_steps/k_point6_cross_replace_river.gif" class="img-fluid" %}

Below, we present the key insights found for the prompt _"A woman's face with blond hair"_.
{% include figure.html path="assets/img/2023-11-07-prompt-to-prompt/analysis/Figure 09 - Results cross replace steps - faces.png" class="img-fluid" %}

Below, we present the key insights found for the prompt _"A river between mountains"_.
{% include figure.html path="assets/img/2023-11-07-prompt-to-prompt/analysis/Figure 10 - Results cross replace steps - landscape.png" class="img-fluid" %}


## A3. Exploration of self-attention hyperparameter ("self replace steps")

In our investigation of the self-attention hyperparameter known as "self_replace_steps," we conducted a series of experiments with careful consideration of the interplay between this parameter and two other critical factors: "k" (the silhouette threshold) and "cross_replace_steps" (the cross-attention injection parameter). To comprehensively assess the influence of "self_replace_steps," we designed two distinct experimental scenarios.

In the first scenario, we set "k" and "cross_replace_steps" to their default values in the literature review (0.3 and 0.8 respectively), creating an environment conducive to exploring the effects of self-attention within these threshold parameters. Concurrently, in the second scenario, we opted for more extreme settings by keeping "k" at 0 (no silhouette threshold) and "cross_replace_steps" at 0.2, thereby intensifying the impact of the self-attention hyperparameter.

### With k = 0.3 and cross_replace_steps = 0.8:
{% include figure.html path="assets/img/2023-11-07-prompt-to-prompt/GIFs/self_replace_steps/k_point3_self_replace_women.gif" class="img-fluid" %}
{% include figure.html path="assets/img/2023-11-07-prompt-to-prompt/GIFs/self_replace_steps/k_point3_self_replace_river.gif" class="img-fluid" %}

### With k = 0 and cross_replace_steps = 0.2:

{% include figure.html path="assets/img/2023-11-07-prompt-to-prompt/GIFs/self_replace_steps/k_zero_crossattention_point2_self_replace_women.gif" class="img-fluid" %}
{% include figure.html path="assets/img/2023-11-07-prompt-to-prompt/GIFs/self_replace_steps/k_zero_crossattention_point2_self_replace_river.gif" class="img-fluid" %}

Below, we present the key insights for the hyperparameter "self_replace_steps" within the context of the prompt _"A woman's face with blond hair"_.
{% include figure.html path="assets/img/2023-11-07-prompt-to-prompt/analysis/Figure 11 - Results self replace steps - faces.png" class="img-fluid" %}

Below, we present the key insights for the hyperparameter "self_replace_steps" found for the prompt _"A river between mountains"_.
{% include figure.html path="assets/img/2023-11-07-prompt-to-prompt/analysis/Figure 12 - Results self replace steps - landscape.png" class="img-fluid" %}

## A4. Cycle Consistency of method

Our primary goal is to delve into the notion of "Cycle Consistency" within our methodology. This concept revolves around the seamless reversal of text prompt modifications back to their original form, ensuring that the resulting image closely mirrors the initial prompt. This bidirectional editing process serves as the central focus of our research, and in the subsequent sections, we present our findings on this crucial aspect.

{% include figure.html path="assets/img/2023-11-07-prompt-to-prompt/analysis/Figure 13 - Cycle consistency.png" class="img-fluid" %}

{% include figure.html path="assets/img/2023-11-07-prompt-to-prompt/analysis/Figure 14 - Cycle consistency - hyperparameter impact.png" class="img-fluid" %}

## B. Generalization of optimized hyperparameters to "attention re-weight method"

After identifying the optimal parameters, we conducted a comparative analysis to assess their generalizability across other methods, including attention re-weighting. In the visual presentation, we used GIFs to showcase image generation under two different parameter configurations for the prompt _"A woman's face with long wavy blond hair"_.

On the left side, images were generated using default values (k=0.3; cross_replace_steps = 0.8; self_replace_steps = 0.2) while varying the assigned weights. Notably, negative weights led to instability and less desirable outcomes, as evidenced by the results on the left.

On the right side, we employed our optimized hyperparameter values (k = 0; cross_replace_steps = 0.2; self_replace_steps = 0.8). These images demonstrated improved stability while consistently producing the desired output. This visual comparison highlights the effectiveness of our optimized parameters and their superior performance, particularly when dealing with attention re-weighting method.

<div style="display: flex;">
  <div style="flex: 1; padding: 10px;">
    Literature suggested parameters
    {% include figure.html path="assets/img/2023-11-07-prompt-to-prompt/GIFs/c_value/c_value_women_curly.gif" class="img-fluid" width="200" %}
  </div>
  <div style="flex: 1; padding: 10px;">
    Newly optimized parameters
    {% include figure.html path="assets/img/2023-11-07-prompt-to-prompt/GIFs/c_value/c_value_women_curly_improved_self_replace.gif" class="img-fluid" width="50" %}
  </div>
</div>


## Our Proposed Method

As our research has demonstrated, the current prompt-to-prompt method, as reported in the literature <d-cite key="hertz2022prompttoprompt"></d-cite>, exhibits significant limitations. Specifically, with the current settings for the silhouette, cross-attention injection, and self-attention injection parameters, the method fails to perform the prompted edits with precision. A comparative analysis of the generated target images against the geometry of the reference images reveals undesired deviations. The existing method over-constrains the geometry due to excessively high k values and cross-attention injection values. Additionally, it underutilizes self-attention injection. Furthermore, the current method lacks cycle consistency.
To address these shortcomings, we propose a new framework: the _“CL P2P”_ prompt-to-prompt image editing framework. This framework offers several key improvements over the existing method:

**Optimization of Critical Hyperparameters**: Our research indicates that optimizing the values of critical hyperparameters results in higher prompt-to-prompt image editing precision and a more accurate similarity between the reference and target images for desired features. We propose the following adjusted values, particularly for editing faces and hairstyles:
* Local editing (silhouette parameter k): 0.0
* Cross-attention injection (cross replace steps): 0.2
* Self-attention injections (self-replace steps): 0.8

{% include figure.html path="assets/img/2023-11-07-prompt-to-prompt/analysis/Figure 16 - Current vs new method comparision of output.png" class="img-fluid" %}

By selecting these values, the following changes are introduced to the prompt-to-prompt editing method:
* <span style="color:red">Remove</span>: Local editing can be removed from the method, as it did not lead to significant improvements compared to the precision achieved by the elongated injection of self-attention.
* <span style="color:orange">Reduce</span>: The cross-attention (query-key-value attention) injection should be reduced to allow greater geometric adaptability and better convergence between the reference and target images.
* <span style="color:green">Increase</span>: Self-attention injection should be substantially elongated from 20% to 80% of the diffusion steps. This is crucial, especially for editing hairstyles, as it allows for the greatest geometric adaptability and ensures the convergence between desired reference and target image features.

{% include figure.html path="assets/img/2023-11-07-prompt-to-prompt/analysis/Figure 15 - Current vs new method.png" class="img-fluid" %}

**Addressing Cycle-Inconsistency**: To remedy the cycle-inconsistency, we propose balancing the asymmetry of the current method with regards to the V values of the underlying transformer model. The current method is cycle-inconsistent, even though the same embeddings are used for both the reference and target prompts. Traditionally, the method has only employed the V values of the reference prompt, neglecting those of the target prompt. This characteristic likely introduces asymmetry, breaking the cycle-consistency of the model. We propose an additional injection mechanism for the “CL P2P” framework, a V value injection method, allowing for the consideration of both the V values of the reference and target images. To control the number of injection steps, we introduce an additional hyperparameter, “V value injection steps”. The V value injection function is defined based on the logic highlighted in the footnote of the image.

## Future work

The development of the “CL P2P” framework is a significant advancement in prompt-to-prompt image editing methods. However, there are still areas where further research will be needed. A critical area of exploration lies in the enhancement of cycle-consistency within the prompt-to-prompt editing process. Further research is required to ascertain and refine the optimal values for the V value injection steps, a key component in achieving cycle-consistency.

Additionally, the existing frameworks predominantly focus on singular reference and target prompts. While this approach has opened new pathways in human-computer interaction, several research questions remain unexplored. A notable inquiry is the potential to integrate various prompt-to-prompt editing methods, such as "word swap", "attention re-weighting," and "prompt refinement." This integration aims to facilitate a dynamic, conversational interaction between users and generated images, enabling a continuous and iterative editing process. Current state-of-the-art generative image models, such as mid-journey models, do not inherently support such iterative mechanisms. The realization of this functionality necessitates extensive research and development, offering an exciting challenge for future advancements in the field.

## Conclusion

Image generation models, inherently stochastic in nature, exhibit variability in outcomes even when similar prompts are applied. This stochasticity can result in significant deviations in the generated images. For instance, prompts like “A woman’s face with blond hair” and “A woman’s face with red hair” may yield images with markedly different facial features, demonstrating the algorithm's underlying randomness.

In response to this challenge, prompt-to-prompt image generation and editing techniques have emerged as a significant area of interest in recent years. These methods, while constituting a potent tool in the arsenal of image editing alongside fine-tuning, semantic mixing, and masking approaches, are not without limitations. Specifically, the precision of edits and the geometric alignment between reference and target images often fall short of expectations.

Our research delves into the influence of critical hyperparameters on the outcomes of a cross-attention-based prompt-to-prompt method. We aimed to dissect the impact of each hyperparameter on image editing and geometric adaptation between the reference and target images. Our findings make substantive contributions to enhancing the precision and geometric convergence in prompt-to-prompt methods, with the following key insights:
* An extensive analysis of three critical hyperparameters (silhouette selection, cross-attention injection, and self-attention injection) was conducted, focusing on their effect on the precision of an attention-based prompt-to-prompt editing method.
* Contrary to existing literature<d-cite key="hertz2022prompttoprompt"></d-cite>, our study reveals that self-attention injection plays a more pivotal role than previously recognized. We recommend incorporating self-attention injection from the reference image for approximately 80% of the diffusion steps during the target image generation process.
* We introduce the novel _“CL P2P”_ framework, designed to elevate the efficacy of prompt-to-prompt editing.

Our research not only deepens the understanding of prompt-to-prompt editing methods but also achieves enhanced editing precision and improved similarity between reference and target images.

Looking ahead, the _“CL P2P”_ framework paves the way for further exploration, particularly in addressing the cycle consistency of prompt-to-prompt methods. Additionally, exploring strategies to seamlessly integrate different prompts into a continuous dialogue could revolutionize human-computer interaction, enabling users to edit generated images through conversational engagement.


