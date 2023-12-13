---
layout: distill
title: Activation Patching in Vision Transformers
# description: 
date: 2023-12-12
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

# Motivation
Neural nets contain large amounts of parameters and connections that they use to model a given phenomenon. Often, the breadth and complexity of these systems make it difficult for humans to understand the mechanisms that the model uses to perform its tasks. The model is treated like a black-box. When attempting to alter the behavior of the model when it does not behave in the desired way, engineers often rely on trial-and-error tuning of hyperparameters or providing larger, more diverse datasets for training. However, it is often difficult to get representative training data. In addtion, hyperparameters can improve training but are limited in their ability to alter the innate limitations of a model.

Mechanistic interpretability aims to unpack the underlying logic and behaviors of neural networks. <d-cite key="zhang2023best"></d-cite> Activation patching is an interpretability technique that replaces activations in a corrupted model with that of an uncorrupted model in order to analyze their influence on model output. When a patched activation improves model performance, it indicates that the patched activation playes a role relevant to the corrupted information. <d-cite key="Vig2020InvestigatingGB"></d-cite> 

A better understanding of the logic within neural networks will allow for more strategic improvements to these models inspired by this newfound understanding. In additon, interpretability is the first step toward changing and correcting models. With an understanding of the underlying mechanisms comes more control of these mechanisms, which can be used to apply necessary changes for goal alignment and mitigating issues such as bias.

# Related Work

Pearl <d-cite key="10.5555/2074022.2074073"></d-cite> defines causal mediation analysis in order to analyze the effect of intermediate entities on a desired result. An application of the "indirect effect" introduced by this research is activation patching, also known as causal tracing. The indirect effect is the effect a given activation has on the output of the model. Since the activation is encompassed within the layers of a neural network, it has an indirect effect on the output. This analysis has been used in language models.<d-cite key="meng2023locating"></d-cite> Here, the indirect effect is defined as the role of an MLP or attention layer on the output. This role is analyzed by first corrupting the outputs of the network. Then, activations from an uncorrupted run of the model can be iteratively patched into the corrupted run in order to determine which activations can best restore the uncorrupted outputs. The activations with the most significant restorative impact have the highest indirect effect.


For example, if the hidden state for a given attention head in a language model with prompt "The Eiffel Tower is in" is patched into that of a prompt "The Colosseum is in" and successfully changes the output from "Rome" to "Paris", this indicates that the patched head contains knowledge about the Eiffel Tower. <d-cite key="meng2023locating"></d-cite> The figure below depicts this process of patching from a clean to corrupt run. 

{% include figure.html path="assets/img/2023-11-10-CNN-activation-patching/patch.png" class="img-fluid" %}<d-cite key="meng2023locating"></d-cite>

Meng et al. also provides an example of how interpretability can open opportunities for model editing. <d-cite key="meng2023locating"></d-cite> With the understanding of where knowledge of facts is stored within the model MLPs, these layers can be edited to edit the knowledge of the language model in a way that is generalizable to many applications of this knowledge within language.

Activation patching has been used for language models, which rely on a transformer architecture. Vision transformers <d-cite key="dosovitskiy2021image"></d-cite> take advantage of the transformer architecture to perform common computer vision tasks such as image classification. The strategies of activation patching can therefore apply in this context as well. Palit et al. performed a similar causal tracing analysis to that of the language model study except with a focus on BLIP, a multi-modal model that can answer questions about a given image. This investigation showed how activation patching can be performed on images along with language rather than language alone.<d-cite key="palit2023visionlanguage"></d-cite>

# Methods

The model that was used for this investigation was a vision transformer that was fine-tuned for the CIFAR10 dataset, a dataset that is often used to train image classification models. The pretrained model that was used, which can be found [here](https://huggingface.co/aaraki/vit-base-patch16-224-in21k-finetuned-cifar10), often fails to classify images in the dataset if they are converted to grayscale. For example, the model classifies the image of a deer below as a cat.
{% include figure.html path="assets/img/2023-11-10-CNN-activation-patching/image.jpg" class="img-fluid" %}{% include figure.html path="assets/img/2023-11-10-CNN-activation-patching/gray.jpg" class="img-fluid" %}
<!-- <img src="assets/img/2023-11-10-CNN-activation-patching/gray.jpg" alt="drawing" style="width:10px;"/> -->

In order to trace which attention heads focus on color information, a clean, corrupted, and restored run was performed with the model. A batch was created was a given image along with a grayscale version of that image. The colored image played the role of the clean run. The grayscale image is a corrupted input that hinders the models ability to classify the object in the image. This is reflected in the logits when the classifier attempts to classify the image. This corrupted run is the baseline in the investigation. Once this baseline was established, the restored run demonstrated the influence of a given attention head. In this run, the hidden state in a given layer was replaced with the hidden state at that layer from the clean run. As demonstrated in previous research <d-cite key="meng2023locating"></d-cite>, a window of layers was restored in order to have an effect on the output, as opposed to just a single layer. In this experiment, the window was 3, so the given layer as well as the adjacent layers were restored. The effect of a given layer was calculated by the difference in the softmax probability of the class of the image between the corrupted and patched run.

{% include figure.html path="assets/img/2023-11-10-CNN-activation-patching/eqn.png" class="img-fluid" %}<d-cite key="meng2023locating"></d-cite>

This analysis was performed for 1000 images from the CIFAR10 dataset. 

# Results

When single layers were patched, results matched that of Meng et al. The patching of a single activation did not have a significan effect on the output. 
{% include figure.html path="assets/img/2023-11-10-CNN-activation-patching/single.png" class="img-fluid" %}

The attention heads of most relevance to color tended to be in the middle or last layers.
{% include figure.html path="assets/img/2023-11-10-CNN-activation-patching/attn.png" class="img-fluid" %}

Here are some examples of tracing for individual images. 
{% include figure.html path="assets/img/2023-11-10-CNN-activation-patching/deer.png" class="img-fluid" %}
{% include figure.html path="assets/img/2023-11-10-CNN-activation-patching/car.png" class="img-fluid" %}
{% include figure.html path="assets/img/2023-11-10-CNN-activation-patching/plane.png" class="img-fluid" %}

The influence of attention heads close to the output align with the conclusions found by Palit et al. This is likely due to direct connection of final layers to the output. There is also a significan influence of middle attention heads on the output, which is some indication of the key information that is stored in these layers. A possible explanation is the closeness of these layers to input layer that directly stores color, but with enough distance from the input to have narrowed down which tokens are relevant to the class the image belongs to. 

# Conclusion

Attention heads have significane in vision transformers not just in the final layers that relate to the output, but in the middle of the network as well. Future investigations could include other forms of corruption, such as by adding noise to the image rather than removing color. This corruption would allow more control on how much the output would change and possibly allow room for more significant restorative effects from patching and more definitive results as to where the most influential attention head live in vision transformers. In addition, performing this analysis with window sizes other than 3 as well as other and larger image datasets could also provide more context as to how important is an individual attention layer.