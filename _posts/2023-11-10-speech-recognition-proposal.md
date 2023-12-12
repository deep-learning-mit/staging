---
layout: distill
title: Modeling Human Speech Recognition with Different Network Architectures
description: Proposes a project evaluating a neural network's ability to effectively model human speech recognition using CNNs vs. TNNs
date: 2023-11-10
htmlwidgets: true

# Anonymize when submitting
# authors:
#   - name: Anonymous

authors:
  - name: Annika Magaro
    affiliations:
      name: MIT

# must be the exact same name as your blogpost
bibliography: 2023-11-10-speech-recognition-proposal.bib  

# Add a table of contents to your post.
#   - make sure that TOC names match the actual section names
#     for hyperlinks within the post to work correctly.
toc:
  - name: Introduction
  - name: Proposal

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

Recent advances in machine learning have made perception tasks more doable by computers, approaching levels similar to humans. In particular, structuring models biologically and using ecologically realistic training datasets have helped to yield more humanlike results. In the field of speech recognition, models trained under realistic conditions with stimuli structured how sounds are represented in the cochlea, with network layers imitating the processing pipeline in the brain, seem to be successful in performing speech recognition tasks. However, it is unclear whether specific network architectures are more beneficial to learning human speech recognition patterns. In this project, I seek to investigate how different network architectures such as CNNs vs. TNNs affect the ability to recognize speech in a humanlike way.


## Proposal

One facet of more biological models is that they attempt to recreate the structure of the human brain. For auditory models, a useful structure to replicate is the cochlea; these replications are called cochleagrams. Cochleagrams have been used in order to model the ear more effectively, leading to models that imitate auditory perception in a more human-like way. A cochleagram works in a similar way to how the cochlea works in a human. It filters a sound signal through bandpass filters of different frequencies, creating multiple frequency subbands, where the subbands for higher frequencies are wider, like how the cochlea works in the human ear. The amplitudes of the different subbands are then compressed nonlinearly, modeling the compressive nonlinearity of the human cochlea <d-cite key="mcdermott2013"></d-cite> <d-cite key="mcdermott2011"></d-cite>.

A recent application of cochlear models to speech perception is found in Kell’s 2018 paper, where they create a convolutional neural network which replicates human speech recognition <d-cite key="kell2018"></d-cite>. They trained the network to recognize a word in the middle of a 2 second clip, from a possible vocabulary of 587 words.  To imitate how the ear functions, they preprocessed the sound signals into cochleagrams, intended to be a more biologically realistic model of the ear. The activations in different layers of the neural network were able to predict voxel responses in different parts of the brain, revealing that the auditory processing pipeline aligned with layers of the network.

In my project, I aim to investigate the importance of network architecture in the ability to effectively model human speech recognition. I plan to train two models, a convolutional neural network and a transformer, and evaluate model performance on speech recognition tasks inspired by Kell 2018. They will be trained on a dataset containing 2 second speech clips from the Common Voice dataset, with a vocabulary of 800 words, imposed on different background noises taken from the Audio Set dataset <d-cite key="ardila2019"></d-cite> <d-cite key="gemmeke2017"></d-cite>. To evaluate the model, I will compare human vs. CNN vs. TNN performance in different types of background noise, and in a few speech manipulations, such as sped-up/slowed-down speech, whispered speech, and sine wave speech. Both models will preprocess signals into cochleagrams, so this project is intended to discover whether convolutional neural networks or transformers can more effectively model the auditory processing pipeline in the brain. Alternatively, it may show that the specific neural network architecture does not matter and effective modeling is more dependent on the cochleagram preprocessing. 

