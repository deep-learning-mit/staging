---
layout: distill
title: Sentence Embeddings
description: Large models, such as large language or vision models, are typically used to obtain embeddings of data, such as text or images. The embeddings are very rich and encode semantic information about the objects. The embeddings can be then later be used for tasks such as similarity search. However, the cost (both money and environmental) of obtaining the embeddings can be large. Given a dataset, can we query the model at 'very few points' which can later be extrapolated to embeddings for other data without querying the large model again?
date: 2023-11-08
htmlwidgets: true

# Anonymize when submitting
# authors:
#   - name: Anonymous

authors:
  - name: Alor Sahoo
    affiliations:
      name: MIT
  - name: Sebastian Alberdi
    affiliations:
      name: MIT

# must be the exact same name as your blogpost
bibliography: 2023-11-08-sentence-embeddings.bib  

# Add a table of contents to your post.
#   - make sure that TOC names match the actual section names
#     for hyperlinks within the post to work correctly.
toc:
  - name: Project Proposal
    subsections:
    - name: Introduction
    - name: Overview
    - name: Limitations
  - name: Citations
---
## Proposal

### Introduction
Querying general LLMs frequently is often slow and expensive, especially at scale.
Our project investigates student-teacher networks, in which we train a less-accurate, but “cheaper” student model by leveraging the knowledge of a more accurate, but expensive, “teacher” model <d-cite key="teacher"></d-cite>.
Already, these types of architectures have already been applied to generate lightweight but performant student networks for a variety of different purposes (classification, recognition, generation, etc.)
Sentence level embeddings—very large vectors that quantify aspects of its content— are one such data that can be expensive to query from a teacher model.
Among other things, these embeddings are useful for quantifying the similarities of different sentences. 

### Overview

#### Methods

Our project will specifically center on HuggingFace’s [pre-trained sentence transformer library](https://www.sbert.net/docs/pretrained_models.html).
We can approximate a “student” network as the less performant, faster “distiluse-base-multilingual-cased-v2 model” and a “teacher” network as the more performant, slower “all-MiniLM-L12-v2” model.
The primary goal will be to determine what specific architecture works best for mapping “student” network embeddings to “teacher network embeddings. 

We will first use the BOOKSUM dataset from HuggingFace (subject to change) and tokenize the sentence appropriately.
Then, we will train our various architectures on 10% of the data by querying both the student and teacher models.
The remaining 90% of our text dataset is used to test the model’s predictions against the embeddings of the teacher model.
While this ratio of training/testing is very skewed, it is representative of the reality that querying the teacher model is expensive.
We will use another dataset (to be determined) to validate our model afterward.

One obvious metric for our model’s performance is the average reconstruction loss, as measured by Euclidean distance.
Another metric is cosine similarity, which gives information on the angle between vectors and is particularly useful at higher dimensional spaces.

#### Architectures

We plan to investigate the following architectures (subject to change):

1. Multi-Layer Perceptron (MLP): MLPs are a simple baseline model to start with, especially since they are easy to train and are universal approximators (in theory).
2. Self-Attention Layer: This allows the model to consider context more and focus on different parts of the input  more easily than in an MLP, potentially improving performance.
3. Recurrent Neural Nets: RNNs have a weak notion of “memory,” allowing it to create context-aware mappings from one sentence embedding to another.

### Limitations

We acknowledge that our approximation of a student and teacher network are imperfect—especially since our student network was not distilled directly from the teacher one.
Also, if our architecture is too resource intensive, then it doesn’t make sense to query the student model and then apply our model, instead of just querying the teacher model directly.
Nonetheless, our project investigates interesting aspects of training on limited data. 

## Citations

