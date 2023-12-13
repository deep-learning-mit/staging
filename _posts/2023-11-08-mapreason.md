---
layout: distill
title: "Reasoning with Maps: Assessing Spatial Comprehension on Maps in Pre-trained Models"
description: Assessing Spatial Comprehension on Maps in Pre-trained Models
date: 2023-11-8
htmlwidgets: true

# Anonymize when submitting
# authors:
#   - name: Anonymous

authors:
  - name: Abdulrahman Alabdulkareem (arkareem@mit.edu)
  - name: Meshal Alharbi (meshal@mit.edu)


# must be the exact same name as your blogpost
bibliography: 2023-11-08-mapreason.bib  

# Add a table of contents to your post.
#   - make sure that TOC names match the actual section names
#     for hyperlinks within the post to work correctly.
toc:
  - name: Motivation
  - name: Project Outline
  - name: "Benchmark & Dataset"
  - name: Black Box
  - name: Investigating representation
    subsections:
    - name: Representation
    - name: Generation

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



# Motivation:
Humans possess a remarkable ability to intuitively understand and make sense of maps, demonstrating a fundamental capacity for spatial reasoning, even without specific domain knowledge. To illustrate this, consider the following question: Do these two maps represent the same location?

{% include figure.html path="assets/img/2023-11-08-mapreason/Picture2.png" class="img-fluid" %}
{% include figure.html path="assets/img/2023-11-08-mapreason/Picture3.png" class="img-fluid" %}

Answering this query necessitates coregistration, the ability to align two maps by overlaying their significant landmarks or key features. Moreover, humans can go beyond mere alignment; they can tackle complex inquiries that demand aligning maps, extracting pertinent data from each, and integrating this information to provide answers.

Now, do contemporary state-of-the-art (SOTA) machine learning models, pre-trained on vast datasets comprising millions or even billions of images, possess a similar capacity for spatial reasoning? This project is dedicated to probing this question.

# Project Outline:
There are three main stages in this project.

## Benchmark & Dataset:
After conducting a brief review of existing literature, we observed a lack of established benchmarks that could effectively evaluate the central question of our project. As a result, we plan to start this project by constructing a simple dataset/benchmark tailored for assessing map comprehension (e.g., coregistration). Our data collection will be sourced from the aviation domain, where a single location is usually depicted in multiple maps, each with distinct styles and information content (like the images shown above).

Furthermore, as a baseline, we are considering assessing category recognition without involving spatial reasoning. As an illustration, we use a dataset with images of cats and dogs rendered in various artistic styles, where the model's task is to determine whether two images belong to the same category or different categories.

## Black Box:
Treating a pre-trained model as a black box. The first question we plan to investigate is if SOTA multimodal models are already capable (i.e., zero-shot testing without any fine-tuning) of this form of spatial reasoning. For example, models like GPT4V, Clip, VisualBERT, and many others. We anticipate that the answer will likely be negative, especially for complex queries like the following:

{% include figure.html path="assets/img/2023-11-08-mapreason/Picture4.png" class="img-fluid" %}
{% include figure.html path="assets/img/2023-11-08-mapreason/Picture5.png" class="img-fluid" %}


“What is the name of the waypoint on top of Erie County?”

This query would require first identifying the landmarks in each map separately, then aligning the two maps using a shared landmark (“Cory Lawrence” or the shoreline in this example), then finding the location of what is being asked (“Erie County” in the second image in this example), then transform that point to the first map using the established mapping, then finally find and return the name of the waypoint as the answer (“WABOR” in this example).

## Investigating representation:
Investigating the representation/embedding of a pre-trained model. If current models prove to be incapable or inadequate in terms of spatial reasoning capabilities, we plan to investigate why this might be the case by examining their internal representations through multiple approaches:

### Representation: 
We will compute the embedding using SOTA CLIP models available then linearly probe the embedding to see if they can solve our task (i.e., few-shot learning on CLIP representation).
### Generation: 
Can we generate maps from the embedding of CLIP models to learn more about what details they capture and what they fail to capture? 
- Use zero-shot image generation to go from clip embeddings to images.
- Or fine-tune a model using parameter efficient tuning (e.g., ControlNet) to better generate images that match our task.



##########################################################################

https://skyvector.com/

https://www.faa.gov/air_traffic/flight_info/aeronav/digital_products/ 
