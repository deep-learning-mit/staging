---
layout: distill
title: Activation Patching in Vision Transformers
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

<!-- Neural nets contain large amounts of parameters and connections that they use to model a given phenomena. Often, the breadth and complexity of these systems make it difficult for humans to understand the mechanisms that the model uses to perform its tasks. The model is treated like a black-box, often leading to a reliance on trial-and-error and larger, more diverse datasets to alter the behavior of the model when it does not behave in the desired way.

Mechanistic interpretability aims to unpack the underlying logic and behaviors of neural networks. <d-cite key="zhang2023best"></d-cite> Activation patching is an interpretability technique that replaces activations in a model with that of another model in order to analyze its role in the output. When a patched activation changes the output of the first model to match that of the second, it indicates that the patched activation is responsible for that task within the model. <d-cite key="Vig2020InvestigatingGB"></d-cite> For example, if the attention head of a transformer model with prompt "The Eiffel Tower is in" is patched into a model with "The Colosseum is in" and successfully changes the output from "Rome" to "Paris", this indicates that the patched head contains information about the Eiffel Tower. <d-cite key="meng2023locating"></d-cite>

{% include figure.html path="assets/img/2023-11-10-CNN-activation-patching/patch.png" class="img-fluid" %}<d-cite key="meng2023locating"></d-cite>

Activation patching has been used for language models, but this project proposes to apply this technique to a vision transformer. <d-cite key="dosovitskiy2021image"></d-cite> Similar to the strategies of existing language model work, there can be patching with a clean and corrupted dataset. For example, a clean model can be trained on color images and a corrupted one can be trained only with black and white images. If an attention head from the color model is patched into the black and white model such that it can correctly classify images, that will reveal that the head contains information about the color of the image. Similar analysis can be performed for other tasks vision models perform before classification, such as edge detection. In this case, the corrupted model can have blurred images.

Given successful analysis of simpler features of the images, it would be interesting to explore how attribution patching can reveal information about vision models with racial and gender bias. Datasets that focus on emotion and actions may have biases when determining people of which demographic are most likely to make a given expression or perform a given action. The mechanisms behind this bias can be dissected further through activation patching. <d-cite key="Vig2020InvestigatingGB"></d-cite>

CIFAR-100 is a potential dataset that can be used for this study. The images can be blurred or changed to black and white as needed. 

Thank you to Saachi Jain for helping me flesh out this idea.

## Related Work (not alr in biblio)
- Transformer Interpretability Beyond Attention Visualization: https://arxiv.org/pdf/2012.09838.pdf
- Causal Explanation of Vision Transformers: https://arxiv.org/pdf/2211.03064.pdf
- Causal Tracing Tool for BLIP (image to text caption generation?): https://arxiv.org/pdf/2308.14179.pdf
- CNN model editing, fossils: https://openaccess.thecvf.com/content/CVPR2023W/L3D-IVU/html/Panigrahi_Improving_Data-Efficient_Fossil_Segmentation_via_Model_Editing_CVPRW_2023_paper.html
- text to image diffusion model editing: https://arxiv.org/abs/2305.16225
- interpreting vision transformer's latent tokens with natural language: https://arxiv.org/abs/2310.10591
- edits on text-to-image projections, diffusion models: https://arxiv.org/abs/2308.14761
- edit generative model: https://arxiv.org/pdf/2007.15646.pdf

## Applications/ Ideas
- fixing: changing knowledge in LM can correct misinfo, bias; what are advantages to changing images?
- train on just color detection, like the shapes dataset

- which attention heads are doing what
  - corrupt embedd: where relevant heads are in network, patch into corrupt network

- corrupt embeddings
- think of applications of that
- patch in activation heads until color is restored

## backup
- augment gender bias or corruption study rather than doing something completely diff

 An introduction or motivation.
2) Background and related work with literature cited.
3) A description of your methods and experiments with figures showing the method or setup.
4) An analysis of the results of your experiments with figures showing the results.
5) A conclusion or discussion, with mention of limitations. -->

# Motivation
Neural nets contain large amounts of parameters and connections that they use to model a given phenomena. Often, the breadth and complexity of these systems make it difficult for humans to understand the mechanisms that the model uses to perform its tasks. The model is treated like a black-box, often leading to a reliance on trial-and-error and larger, more diverse datasets to alter the behavior of the model when it does not behave in the desired way.

Mechanistic interpretability aims to unpack the underlying logic and behaviors of neural networks. <d-cite key="zhang2023best"></d-cite> Activation patching is an interpretability technique that replaces activations in a model with that of another model in order to analyze its role in the output. When a patched activation changes the output of the first model to match that of the second, it indicates that the patched activation is responsible for that task within the model. <d-cite key="Vig2020InvestigatingGB"></d-cite> 

# Related Work

Pearl <d-cite key="10.5555/2074022.2074073"></d-cite> defines causal mediation analysis in order to analyze the role of intermediate entities on an output. An application of the "indirect effect" introduced by this research is activation patching, also known as causal tracing. The indirect effect is the effect a given activation has on the output of the model. This technique has been used in language models <d-cite key="meng2023locating"> where the indirect effect is defined as the role of an MLP or attention layer on the output. This role is analyzed by first corrupting the outputs of the network and determining. Then, states from an uncorrupted version of the network are patched into the corrupted network.


For example, if the attention head of a transformer model with prompt "The Eiffel Tower is in" is patched into a model with "The Colosseum is in" and successfully changes the output from "Rome" to "Paris", this indicates that the patched head contains information about the Eiffel Tower. <d-cite key="meng2023locating"></d-cite>

{% include figure.html path="assets/img/2023-11-10-CNN-activation-patching/patch.png" class="img-fluid" %}<d-cite key="meng2023locating"></d-cite>

Activation patching has been used for language models<d-cite key="dosovitskiy2021image"></d-cite> 

# Methods

The model that was used for this investigation was a vision transformer that was fine-tuned for the CIFAR10 dataset, a dataset that is often used to train image classification models. The pretrained model that was used, which can be found [here](https://huggingface.co/aaraki/vit-base-patch16-224-in21k-finetuned-cifar10), often fails to classify images in the dataset if they are converted to grayscale. For example, the model classifies the image of a deer below as a cat.
{% include figure.html path="assets/img/2023-11-10-CNN-activation-patching/image.jpg" class="img-fluid" %}{% include figure.html path="assets/img/2023-11-10-CNN-activation-patching/gray.jpg" class="img-fluid" %}
<!-- <img src="assets/img/2023-11-10-CNN-activation-patching/gray.jpg" alt="drawing" style="width:10px;"/> -->

In order to trace which attention heads focus on color information, a clean, corrupted, and restored run was performed with the model. A batch was created was a given image along with a grayscale version of that image. The colored image played the role of the clean run. The grayscale image is a corrupted input that hinders the models ability to classify the object in the image. This is reflected in the logits when the classifier attempts to classify the image. This corrupted run is the baseline in the investigation. Once this baseline was established, the restored run demonstrated the influence of a given attention head. In this run, the hidden state in a given layer was replaced with the hidden state at that layer from the clean run. As demonstrated in previous research <d-cite key="meng2023locating"></d-cite>, a window of layers was restored in order to have an effect on the output, as opposed to just a single layer. 

This analysis was performed for 1000 images from the CIFAR10 dataset. 

# Results

# Conclusion