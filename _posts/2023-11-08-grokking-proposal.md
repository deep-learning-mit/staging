---
layout: distill
title: Grokking Proposal
description: What sparks the mysterious ''grokking'' in neural networks-a sudden leap in learning beyond training? This proposal outlines our blog's mission to investigate this perplexing event. We're set to explore the triggers and theories behind grokking, seeking to understand how and why these moments of unexpected intelligence occur.
date: 2023-11-08
htmlwidgets: true

# Anonymize when submitting
# authors:
#   - name: Anonymous

authors:
  - name: Siwakorn Fuangkawinsombut
    affiliations:
      name: MEng 6-3, MIT 
  - name: Thana Somsirivattana
    affiliations:
      name: BS 18 & 6-3, MIT

# must be the exact same name as your blogpost
bibliography: 2023-11-08-grokking-proposal.bib  

# Add a table of contents to your post.
#   - make sure that TOC names match the actual section names
#     for hyperlinks within the post to work correctly.
toc:
  - name: Introduction
  - name: Related Works
  - name: Timeline
    subsections:
    - name: Week 1
    - name: Week 2
    - name: Week 3
    - name: Week 4

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

In the study of neural networks, “grokking” is a phenomenon first observed by (Power et. al. 2022) in which a model trained on algorithmic tasks suddenly generalize long after fitting the training data. The project aims to understand grokking and the conditions that prompt it by (i) experimenting with various data sets and model architectures; (ii) surveying plausible explanations that have been proposed; and (iii) performing further experiments to assess the plausibility of those explanations.

## Related Works

Based on a cursory look at the literature on the topic, we plan to investigate the effects of training size, weight decay, and model complexity on grokking. Our goals are to (i) replicate the grokking phenomenon; (ii) provide some intuitive explanations of the phenomenon, which includes clarifying its relationship to the more well-known “double descent” phenomenon; and (iii) test some of the proposed explanations in the literature.

Some of the relevant papers we plan to look into are:
1. Grokking: Generalization beyond overfitting on small algorithmic datasets<d-cite key="power2022grokking"></d-cite>
2. A Tale of Two Circuits: Grokking as a competition of sparse and dense subnetworks<d-cite key="merrill2023tale"></d-cite>
3. Unifying Grokking and Double Descent<d-cite key="davies2023unifying"></d-cite>
4. Explaining grokking through circuit efficiency<d-cite key="varma2023explaining"></d-cite>
5. Grokking as the Transition from Lazy to Rich Training Dynamics<d-cite key="kumar2023grokking"></d-cite>
6. Progress measures for grokking via mechanistic interpretability<d-cite key="nanda2023progress"></d-cite>
7. To grok or not to grok: Disentangling generalization and memorization on corrupted algorithmic data<d-cite key="doshi2023grok"></d-cite>
8. Grokking Beyond Neural Network: An empirical exploration with model complexity<d-cite key="miller2023grokking"></d-cite>

{% include figure.html path="assets/img/2023-11-08-grokking-proposal/power_plot.png" class="img-fluid" %}
*This figure illustrates the grokking phenomenon in neural networks* <d-cite key="power2022grokking"></d-cite>

## Timeline

### Week 1: Foundation and Replication
* Delve into the literature on grokking.
* Replicate the grokking phenomenon.

### Week 2: Hypothesis and Experimentation
* Formulate hypotheses based on insights from reading the literature.
* Design and conduct targeted experiments.

### Week 3: Testing and Analysis
* Test the proposed hypotheses in varied scenarios.

### Week 4: Synthesis and Reporting
* Compile and synthesize the findings.
* Write the blog post.
