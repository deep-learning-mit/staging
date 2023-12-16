---
layout: distill
title: Understanding Limitations of Vision-Language Models
date: 2022-12-01
htmlwidgets: true



# Anonymize when submitting
# authors:
#   - name: Anonymous

authors:
  - name: Shelley Choi
    affiliations:
      name: MIT
  - name: Siddharth Somasundaram
    affiliations:
      name: MIT

# must be the exact same name as your blogpost
bibliography: 2022-12-01-distill-example.bib  

# Add a table of contents to your post.
#   - make sure that TOC names match the actual section names
#     for hyperlinks within the post to work correctly.
toc:
  - name: Initial Prompt
  - name: Proposal Overview
  - name: Potential Research Questions
    subsections:
    - name: Bias to Text Labels
    - name: Transfer Learning
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

## Initial Prompt 

Joint vision/language models such as CLIP try to align vision and language latent spaces. This provides an extra level of visibility into the representations: for example, for a given image of a cat, its similarity to the text embedding of "a photo of a cat" typically captures how "cat-like" the image is. This project would involve studying the representation space of such models with respect to sensitive attributes/biases. For example, given photos of either men or women, which image embeddings are closer to the caption "a photo of a firefighter." This project would involve performing a systematic study to identify biases in the representations of such models. 

## Proposal Overview

The idea behind the project is to explore joint vision/language models that try to align vision and language latent spaces. In that search, we take a closer look at OpenAI’s Contrastive Language-Image Pre-training (CLIP) [1] released in Feb 2021 and Wayve’s GAIA-1 [2] introduced in June 2023. CLIP consists of a convolutional neural network that transforms an image, and a transformer neural network that transforms text. These networks use contrastive modeling to compare similarity between the image and text space, and its zero-shot learning capabilities allow generalization across a variety of new concepts. GAIA can generate videos of driving simulations from a variety of inputs such as video, text, and/or action inputs. These inputs are then encoded into a common representation of tokens that are fed into a transformer (world model) that predicts the next image tokens.

Regarding this topic, we had several ideas for research questions. Based on instructor feedback, we’re hoping to focus on one of them for the final project.


## Potential Research Questions
### Idea #1: Investigating and Mitigating Bias to Text Labels

The first idea we were thinking of is related to contrastive learning with augmentations in label space instead of input space. The goal of contrastive learning is to ensure a constant output with respect to certain variations in the input. We note that vision-language models (e.g. GAIA, CLIP) are trained with text labels for the image inputs. However, a single text description is not a unique identifier of an image; there are many possible descriptions of a single image. For example, the text label of an image might take the form “Dad sitting on the couch”. An equally valid, but different, text label would be “A person napping on the sofa”. How would vision-language models handle these different cases?

*Scientific Question: Can augmentations in label space allow GAIA, CLIP, etc. to learn better representations with fewer data points?*

- Will the text encoder map each of these two texts to similar latent spaces? 
- How would downstream task performance be affected by using multiple label augmentations? 
- If performance improves, could label augmentations enable training and convergence with fewer data samples?

*Possible Outcomes*
- Either these models learn representations that map multiple labels to similar points in feature space, or
- the choice of text label affects how features in image space are encoded

### Idea 2: Addressing Limitations via Transfer Learning
We also wanted to ask: How can multi-modal generative AI models trained on a specific dataset be generalized and decrease bias? GAIA, in particular, was specifically trained using Wayve’s UK urban driving data. In the UK, drivers drive on the left hand side of the road. Furthermore, the dataset primarily focuses on urban roads, where there are clearly defined lines that indicate asphalt concrete roads. We want to see if this model can also be applied to countries that don’t necessarily follow these “rules” that the GAIA model learned. Can the model also discover other “rules” where vehicles drive on the right side of the road in other parts of the world, or where roads do not have clear structure in less developed countries? 
 
GAIA unfortunately does not publish its data, so we cannot know whether the model truly achieves data symmetry. However, we could take the following approaches in transfer learning, where we can likely reuse the GAIA model and generalize to other places with different rules. Alternative options or further details will likely come as we learn more about transfer learning in class during Week 11. 

*Approach 1: Dual-encoder contrastive learning*

Dual-encoder contrastive learning, which is part of the contrastive learning that maximizes the similarity between similar items and minimizes the similarity between dissimilar items, allows consideration of two different data domains.
We define dual-encoder contrastive loss to be the following, where the two data domains $$\chi_1$$ and $$\chi_2$$ represent images and text, respectively. The encoder $$f_1$$ can map images to a fixed-dimensional space using convolutional neural networks (CNN), and the encoder $$f_2$$ can map text using a transformer:


After training, a decoder can take in the image and text embeddings to generate a series of images $$V_i$$ that constitute a video $$V$$. Once we learn the meaningful representations of the multimodal input data that can be mapped onto a singular space, it becomes easier to understand their relationship to aid in domain adaptation—we can utilize a similar multi-modal structure. 


*Approach 2: Few-shot learning*

Few-shot learning helps the model to recognize and evaluate situations where there may be sparse data. It would address GAIA’s lack of diverse data. For example, it would allow GAIA to be expanded to images from other countries (that may have more side roads or undefined roads) to text that describes situations that are rarely encountered in the UK (extreme weather situations such as a tornado) without having extensive labeled data.
Once we are able to capture the relationships between the different domains, where we can identify potential “base classes,” we can use that information for few-shot learning and achieve good generalization for GAIA. Some techniques might involve recurrent neural networks (RNN) or siamese networks.

## References
1. Radford et al., *“Learning transferable visual models from natural language supervision”*, ICML 2021
2. Hu et al., *“GAIA-1: A Generative World Model for Autonomous Driving”*, arXiv 2023

