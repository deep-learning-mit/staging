---
layout: distill
title: Transfer Resistant Model Training
description: This blog post details our work on training neural networks that
  are resistant to transfer learning techniques.
date: 2023-12-12
htmlwidgets: true

# Anonymize when submitting
# authors:
#   - name: Anonymous

authors:
  - name: Ryan Yang
    url: "https://www.google.com/url?sa=i&url=https%3A%2F%2Fmanipulation.csail.mit.edu%2FFall2023%2Findex.html&psig=AOvVaw3MuJLCZwr7MxMiaaFQTBeC&ust=1699601771753000&source=images&cd=vfe&opi=89978449&ved=0CBIQjRxqFwoTCNil45C0toIDFQAAAAAdAAAAABAH"
    affiliations:
      name: MIT
  - name: Evan Seeyave
    url: ""
    affiliations:
      name: MIT

# must be the exact same name as your blogpost
bibliography: 2023-11-09-transfer-resistant-model-training.bib

# Add a table of contents to your post.
#   - make sure that TOC names match the actual section names
#     for hyperlinks within the post to work correctly.
toc:
  - name: Introduction and Motivation
  - name: Related Works
  - name: Methods
  - name: Experiments
  - name: Results
  - name: Analysis
  - name: Conclusion
---

## Introduction and Motivation

{% include figure.html path="assets/img/2023-11-09-transfer-resistant-model-training/teacher_student.gif" class="img-fluid" %}
We are interested in robustness of models against fine-tuning or transfer learning. The motivating example is as follows: suppose there is a model trained to be capable of classifying a dataset. An external agent wants to train a model to classify a different dataset for a possibly malicious purpose. With transfer learning, this is possible and performs well by replacing and retraining just the last few model layers <d-cite key="zhuang2020comprehensive"></d-cite>. We aim to investigate a method of training the model to be capable of classifying the original set of classes but is more difficult to transfer to different datasets. Thus we aim to answer the question: How can we train a model such that it is robust to transfer learning on a new dataset?

## Related Works

The authors are not aware of previous work in the realm of improving robustness of models against transferability. There have been previous analyses of transfer learning, most commonly found in convolutional neural networks <d-cite key="zhuang2020comprehensive"></d-cite>.
A related problem is machine unlearning which takes a trained model and attempts to make the model forget defined points of information <d-cite key="cao2015towards"></d-cite>. However, our problem setting is different in that we wish to prevent learning undesirable pieces of information from the beginning of training as opposed to forgetting after training.

## Methods

learning $$\theta$$ that 

## Experiments

The problem settings above relating to transfer learning and machine unlearning often involve large convolutional neural networks (CNNs) or language models. Due to computational constraints, this will not be feasible for this project. Rather, we will investigate a toy problem setting.
The toy setting will focus on a shallow CNN with the MNIST dataset. We will split the MNIST dataset into two sets, a “desirable” set and “undesirable” set. For example, the “desirable” set contains images with labels from 0 to 4. The undesirable set will contain all images with labels from 5 to 9. We aim to train a CNN that successfully classifies the images in the “desirable” set but is difficult to then be trained on the “undesirable” set. Specifically, we aim to find an intervention to training on the “desirable” set such that replacing and retraining the last layer of the CNN for the “undesirable” set, takes longer than replacing and retraining the last layer of a CNN without any intervention. Note that for our problem setting, we assume we have access to samples and classes in the “undesirable” set when training with an intervention on the “desirable” set.

## Results

## Analysis

The most straightforward benchmark is the performance of the model with the intervention versus the model without the intervention after transferring to the “undesirable” set. Our objective is that the performance of the model with the intervention on the “undesirable” set is significantly worse than the model without the intervention. Qualitatively, we aim to provide figures of features learned by the CNN with the intervention and without the intervention. Specifically, we hope to show some features learned in the CNN with intervention are qualitatively different from the features learned in the CNN without intervention using methods such as Grad-CAM <d-cite key="selvaraju2017grad"></d-cite>.

## Conclusion
