---
layout: distill
title: Foley-to-video generating video from environmental audio
description: In this blog we will explore the optimal architecture for generating video from environmental sound inputs.
date: 2022-11-08
htmlwidgets: true

# Anonymize when submitting
# authors:
#   - name: Anonymous

authors:
  - name: Esteban Ramirez Echavarria
    url: "https://www.linkedin.com/in/esteban-raech/"
    affiliations:
      name: LGO, MIT
  - name: Arun Alejandro Varma
    url: "https://www.linkedin.com/in/arunalejandro/"
    affiliations:
      name: LGO, MIT

# must be the exact same name as your blogpost
bibliography: 2023-11-08-foley-to-video.bib  

# Add a table of contents to your post.
#   - make sure that TOC names match the actual section names
#     for hyperlinks within the post to work correctly.
toc:
  - name: Background
  - name: Objective
  - name: Plan
  - name: Bibliography

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

## Background

In filmmaking, a “Foley” is the “reproduction of everyday sound effects that are added to films in post-production to enhance audio quality.” Conversely, we aim to produce what we have dubbed (no pun intended) “Antifoleys” – the reproduction of video that could feasibly have accompanied the inputted audio. Below we discuss our plan of action, along with conceptual and technical questions we will explore.

## Objective
A lot of research has been done in music generation and audio detection, as well as text-to-video and video generative models. The goal of this project is to leverage existing data and models, and explore a novel application of these models when working together.

## Plan
The success of our project depends on accomplishing two things: identifying a successful architecture and gathering the right data. To achieve these, we ask ourselves guiding conceptual questions. We do not have answers to all of them yet – it is an ongoing discussion.

* What will be our inputs and outputs? We plan to separate video and audio channels from the same data, then use audio as inputs and video as outputs. 
* What type of data preprocessing is necessary? We will use the harmonic representation of the input and compare it to the raw waveform of the audio input. This may yield more promising encodings for the audio. 
* Does our model need a temporal sense / sequential embedding? On one hand, perhaps not – at minimum, we simply need a 1:1 mapping between each second of video and audio. On the other hand, if we want the video to seem coherent, our model probably does need a sense of sequentiality. This will help determine the architectures we select.
* Do we leverage an existing model and apply transfer learning? Do we build it from scratch?
* What architectures lend itself well to this task? Since we are associating two different forms of data, representation learning might be a strong candidate. We have considered a form of Autoencoder, where the encoder encodes the audio and the decoder decodes to video.
* Where do we find our data?
* Where do we find existing models?


## Bibliography
1. ACOUSTIC SCENE CLASSIFICATION: AN OVERVIEW OF DCASE 2017 CHALLENGE ENTRIES
2. Data-Driven Harmonic Filters for Audio Representation Learning
3. Conditional GAN with Discriminative Filter Generation for Text-to-Video Synthesis