---
layout: distill
title: Better ASR for Low-Resource Languages using Transfer Learning
description: Project Proposal
date: 2023-11-07
htmlwidgets: true

# Anonymize when submitting
# authors:
#   - name: Anonymous

authors:
  - name: Ryan Conti
  - name: William Wang

# must be the exact same name as your blogpost
bibliography: 2023-11-07-distill-example.bib

# Add a table of contents to your post.
#   - make sure that TOC names match the actual section names
#     for hyperlinks within the post to work correctly.
toc:
  - name: Introduction
  - name: Project Outline
  - name: Goals

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

Automatic speech recognition (ASR) systems have made significant improvements in accurately transcribing spoken language among highly-resourced languages, and have been steadily growing over the past few years. Training modern state-of-the-art ASR systems requires fitting over a large amount of data. Thus, high-resource languages such as English and Spanish, for which labelled data is plentiful, ASR systems have flourised. On the other hand, performance on low resource languages, which comprise most of the world's languages, remains considerably worse due to the lack of sufficient annotated audio for training. Among the many possible approaches to solve this problem, this project will examine ways to improve current approaches for ASR in low resource settings by leveraging the large amount of annotated data available in high-resource languages.

In the past few years, there has been considerable work put into cross-lingual learning in ASR systems. Conneau et al demonstrated that model performance significantly improved when using unlabelled cross-lingual data before finetuning with labelled data <d-cite key='conneau2020unsupervised'><d-cite>, and a very recent study from Google by Zhang et al. pushed the boundaries of this technique, training a model over a large unlabelled dataset spanning over 300 languages <d-cite key='zhang2023google'><d-cite>. Zhang et al. also noted that this pretraining step allowed for the model to produce state of the art results after fine-tuning, despite only using a fraction of the amount of labelled data as previous SotA models, and was even able to perform well on low-resource languages for which it had not seen any labelled data <d-cite key='zhang2023google'><d-cite>.

In this study, we will see if the effects observed by Zhang et al. can be replicated without having to train such a universal ASR system using so much data. In particular, we isolate the objective of high performance on low-resource languages, and investigate whether pre-training a smaller model on high-resource languages which are phonetically similar to a target low-resource language can improve performance on the target low-resource language. We will also investigate the effects of the amount of data required from the low-resource language and the efficacy of the cross-lingual pre-training as a function of phonetic similarity between the two languages. Finally, as a potential last investigation, we will examine the effects of supporting the model's performance on the low-resource language by encoding varying amounts of linguistic knowledge in a weighted finite state transducer (WFST).

## Project Outline

We will approach this task in the following manner:

1. First, we will select languages to train on. Ideally, this will consist of multiple high-resource languages with varying similarities to a target low-resource language. Unfortunately, because of ethical concerns often associated with sourcing in low-resource languages, this may not be possible, and we instead defer to choosing a high-resource language as a target language, but restrict the amount of labelled data we can use. This has the added benefit of being able to control the amount of data more flexibly.

2. We will do data collection and aggregation in the form of annotated audio data for all chosen languages. This will also involve producing the smaller datasets for the target simulated low-resource language.

3. We will choose our model and pre-train on the high-resource languages. There is a wealth of models in the literature, so we haven't exactly decided the best one to suit this project, though the cross-lingual model proposed by Conneau et al <d-cite key='conneau2020unsupervised'><d-cite> seems viable to use.

4. We will then finetune the ASR on the target simulated low-resource language and compare performance with different pre-training methods (including monolingual training with only the low-resource language and, time-permitting, using weighted finite state transducers (WFSTs) to encode various levels of linguistic rules into the training of the low-resource language), labelled target dataset sizes.

## Goals

Through this project, we seek to answer the following questions, among any other interesting questions that arise during our process:

What is the relationship between phonetic and phonemic similarity of high-resource languages and the target language and the effectiveness of the ASR model? In what ways does this transfer learning improve low-resource language ASR models? In what ways, if any, can this transfer learning adversarially impact model performance? How does encoding target-language phonological knowledge in the form of WFSTs affect the performance of the model on the target language?
