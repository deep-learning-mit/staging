---
layout: distill
title: "Tracing the Seeds of Conflict: Advanced Semantic Parsing Techniques for Causality Detection in News Texts"
description: This blog post outlines a research project aiming to uncover cause-effect-relationships in the sphere of (political) conflicts using a frame-semantic parser.
date: 2023-11-09
htmlwidgets: true

# Anonymize when submitting
# authors:
#   - name: Anonymous

authors:
  - name: Philipp Zimmer
    url: "https://www.linkedin.com/in/pzimmer98mit/"
    affiliations:
      name: IDSS, Massachusetts Institute of Technology

# must be the exact same name as your blogpost
bibliography: 2023-11-09-conflict-causality.bib  

# Add a table of contents to your post.
#   - make sure that TOC names match the actual section names
#     for hyperlinks within the post to work correctly.
toc:
  - name: Introduction
  - name: Literature Background
    subsections:
    - name: Qualitative Research on Conflicts
    - name: The Role of Quantitative Methods
    - name: Bridging the Gap with Explainable Modeling Approaches
  - name: Data
  - name: Proposed Methodology
  - name: Timeline
  - name: Outlook

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

This project proposes a novel approach to the study of (political) conflicts by adapting and finetuning an RNN-based frame-semantic parser, as introduced by <d-cite key="swayamdipta2017frame"></d-cite>. 
The goal is to bridge the gap between quantitative and qualitative conflict research methodologies. 
By extracting and analyzing causal relationships from newspaper articles, this project aims to enhance our understanding of conflict dynamics and contribute to more effective conflict prediction and prevention strategies.


## Literature Background

### Qualitative Research on Conflicts

Qualitative research has long been a cornerstone in the study of political conflicts. 
This body of work, now well-established, emphasizes the unique nature of each conflict, advocating for a nuanced, context-specific approach to understanding the drivers and dynamics of conflicts. 
Researchers in this domain have developed a robust understanding of the various pathways that lead to conflicts, highlighting the importance of cultural, historical, and socio-political factors in shaping these trajectories. 
While rich in detail and depth, this approach often faces challenges in scalability and systematic analysis across diverse conflict scenarios.

### The Role of Quantitative Methods

In contrast, the advent of computational tools has spurred a growing interest in quantitative approaches to conflict research. 
These methods primarily focus on predicting the severity and outcomes of ongoing conflicts, with some success. 
However, the onset of conflicts remains challenging to predict, indicating a need for more sophisticated tools and methodologies. 
While offering scalability and objectivity, the quantitative approach often struggles to capture the intricate nuances and evolving nature of conflicts, a gap that qualitative research addresses.

### Bridging the Gap with Explainable Modeling Approaches

The challenge now lies in bridging the insights from qualitative research with the systematic, data-driven approaches of quantitative methods. 
While the former provides a deep understanding of conflict pathways, the latter offers tools for large-scale analysis and prediction. 
The key to unlocking this synergy lies in developing advanced computational methods to see the smoke before the fire – identifying the early precursors and subtle indicators of impending conflicts.


## Data

The project capitalizes on the premise that risk factors triggering a conflict, including food crises, are frequently mentioned in on-the-ground news reports before being reflected in traditional risk indicators, which can often be incomplete, delayed, or outdated. 
By harnessing newspaper articles as a key data source, this initiative aims to identify these causal precursors more timely and accurately than conventional methods. 
We source the analyzed articles from [NewsAPI](https://newsapi.org/), which provides an extensive and constantly updated collection of journalistic content. 
This approach ensures a rich and diverse dataset, crucial for effectively training and testing the model in capturing a broad spectrum of conflict indicators.


## Proposed Methodology

Building on the work by <d-cite key="swayamdipta2017frame"></d-cite>, this project aims to adapt the frame-semantic parser to focus on the nuances of causal relationship identification in the context of conflicts. 
We commence by carefully selecting a set of seed phrases and terms related to conflict. 
The selection is based on relevant terminology extracted from the rigorous past qualitative research work mentioned above. 
Next, we will narrow down to the final seed selection by testing the candidate seeds' semantic similarity to the term conflict.
The resulting set will act as the list of "effects" that we are trying to identify with the frame-semantic parser. 

With regards to the model, we finetune the frame-semantic parser infrastructure with a few-shot learning of conflict-related cause-effect relations. 
We will also experiment with changes of the existing model architecture (incl. data augmentation of the news articles, an additional embedding layer focused on conflict-related content and switching the RNN-base to an LSTM-base). 
Then, the frame-semantic parser will be utilized to extract semantic causes of conflicts appearing in the same frame as one of the selected seeds. 
Frames lacking at least one "cause" and one "effect" will be discarded, as are frames in which the "effect" constituents do not contain any seed key phrase related to conflict. 
An ultimate verification step involves running the Granger causality test to check which identified causes are statistically significant.


## Timeline

* November 14th: Finish data collection
* November 28th: Evaluate the performance of the vanilla implementation of the parser by <d-cite key="swayamdipta2017frame"></d-cite> and test changes to the model architecture.
* December 5th: Optimize the final model design's performance and visualize findings.
* December 12th: Submission of final blog post


## Outlook

By combining advanced NLP techniques with deep theoretical insights from conflict research, this project offers a transformative approach to understanding conflicts. 
The successful adaptation and finetuning of the frame-semantic parser promise not only a technical advancement in semantic parsing of news articles – an emerging novel data source – but also a significant step forward for the field of conflict research.
