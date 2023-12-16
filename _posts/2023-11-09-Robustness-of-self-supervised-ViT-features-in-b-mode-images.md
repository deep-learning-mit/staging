---
layout: distill
title: Robustness of self supervised ViT features in b-mode images
description: Project proposal for 6.S898 Deep Learning MIT class
date: 2023-11-09
htmlwidgets: true

# Anonymize when submitting
# authors:
#   - name: Roger Pallares Lopez

authors:
  - name: Roger Pallares Lopez
    url: "https://www.linkedin.com/in/rogerpallareslopez/"
    affiliations:
      name: Mechanical Engineering Department, MIT


# must be the exact same name as your blogpost
bibliography: 2023-11-09-Robustness-of-self-supervised-ViT-features-in-b-mode-images.bib

# Add a table of contents to your post.
toc:
  - name: Introduction
  - name: Project Description
---

## Introduction
B-mode ultrasound imaging is a widely employed medical imaging technique that uses high-frequency sound waves to
produce visual representations of the internal structures of the human body. Its main advantages are its ability
to produce real-time images, its portability, low cost, and especially the fact that is noninvasive and safe
(non-radiating). However, it is an imaging modality that carries a very high noise-to-signal ratio. Speckle noise,
out-of-plane movement, and high variability in image reconstruction across devices make the resulting images complex
to interpret and diagnose <d-cite key="us"></d-cite>. As an example, the following image shows a b-mode ultrasound image.

{% include figure.html path="assets/img/2023-11-09-Robustness-of-self-supervised-ViT-features-in-b-mode-images/img1.png" class="img-fluid" %}
<div class="caption">
  Ultrasound b-mode image of the upper arm with the main physiology annotated.
</div>

Self-supervised Vision Transformers (ViT) have emerged as a powerful tool to extract deep features for a variety of
downstream tasks, such as classification, segmentation, or image correspondence. Especially, DINO architectures <d-cite key="dino1"></d-cite> <d-cite key="dino2"></d-cite>
have exhibited striking properties, where its features present localized semantic information shared across related
object categories, even in zero-shot methodologies <d-cite key="dino_feat"></d-cite>. Consequently, the aforementioned properties of DINO may allow
us to develop efficient yet simple methods for b-mode ultrasound image interpretation, without the need for an expert
or ground truth labels.

{% include figure.html path="assets/img/2023-11-09-Robustness-of-self-supervised-ViT-features-in-b-mode-images/img2.png" class="img-fluid" %}
<div class="caption">
  DINOv2 segmentation of different objects. Note the consistency between parts of real vs toy/drawn objects of the same category. Adapted from <d-cite key="dino2"></d-cite>.
</div>

## Project Description

We propose analyzing the performance and robustness of DINO in b-mode ultrasound images of the upper and lower limbs.
We note that this dataset features a set of images with a high noise-to-signal ratio, which is a property that DINO
has not yet been tested against. In particular, we will focus on assessing DINO in segmentation and correspondence
tasks in a zero-shot approach. We will perform so by applying dimensionality reduction algorithms and subsequent
clustering to the deep features of the model.

For the segmentation task, we will try to segment bone and fascia tissues from arm images obtained from a subject
while is moving. For the correspondence task, we will try to find correspondence between bones and fascia of images
from 4 different sources:  arm (subject 1 device 1), arm (subject 2 device 1), arm (subject 1 device 2), and leg
(subject 1 device 2).

{% include figure.html path="assets/img/2023-11-09-Robustness-of-self-supervised-ViT-features-in-b-mode-images/img3.png" class="img-fluid" %}
<div class="caption">
  Example of one image of each source. A) Labeled bone and fascia. B) Arm (subject 1 device 1). C) Arm (subject 2 device 1). D) Arm (subject 1 device 2). E) Leg (subject 1 device 2)
</div>
In addition, we aim to explore how these features change from a shallower to a deeper layer, trying to understand
what positional and semantic information they carry. Finally, to further test and challenge DINO in an even more
unfavorable scenario, we will gradually include adversarial noise in our dataset, assessing how the performance
changes.

In order to assess the efficacy of the model in all the aforementioned tasks and tests, both qualitative and
quantitative methods will be employed. Qualitatively, we will plot clusters and segmented images. Quantitatively,
we will label bone and fascia in images from the presented 4 sources and compute accuracy, Dice, and IoU metrics.
Through all these experiments, we hope to gain insights into the feasibility of implementing DINO models in real-world
medical imaging applications.
