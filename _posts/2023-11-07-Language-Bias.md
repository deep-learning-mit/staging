---
layout: distill
title: Understanding Bias in Language Models
description: Do language models have biases that make them better for latin based languages like English? 
date: 2023-11-07
htmlwidgets: true

# Anonymize when submitting
# authors:
#   - name: Anonymous

authors:
  - name: Shreya Karpoor
    url: 
    affiliations:
      name: MIT
  - name: Arun Wongprommoon
    url: 
    affiliations:
      name: MIT

# must be the exact same name as your blogpost
bibliography: 2022-12-01-distill-example.bib  

# Add a table of contents to your post.
#   - make sure that TOC names match the actual section names
#     for hyperlinks within the post to work correctly.
toc:
  - name: Overview
  - name: Research Questions
    subsections:
    - name: Experimental Design
    - name: Exploring Preprocessing
  - name: Citations

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

## Overview

One of the most popular domains for machine learning is for processing audio, with tasks such as automatic speech recognition being the forefront problems still to this day. For clean audio in English, the problem seems to have been solved, but accuracy seems to deteriorate for other languages. Currently the most popular machine learning models used for this task are RNNs and Transformers, which are specifically designed to process data on a time series.

Time series data, however, might not be as simple as in English. The motivation for this project stems from the team’s second languages, and how their writing systems are not simply letter-by-letter from first to last. We are hypothesizing that out-of-order label classification problems challenge models and expose their structural biases.

{% include figure.html path="assets/img/2023-11-07-Language-Bias/experiment_drawing.png" class="img-fluid" %}

### Research Questions

1. How do models like RNNs and Transformers learn out-of-order label classification (which is the basis for speech and language tasks)?
2. More specifically, is there a structural bias that makes transformers better suited to Latin based languages like English?

For reference, there are features in different languages’ writing that may complicate model accuracy. How can we characterize how each of these features affects model accuracy?
- English writing is prevalent with silent letters like in knight, vegetable, and idiosyncrasies
- Arabic writing omits vowels (kitab is written as ktb)
- Thai and other southeast asian writing place vowels out of order (e+r+i+y+n spells rieyn)
- Looking more broadly, in terms of word order, In Sanskrit, word order does not matter (i.e. food eating elephants = elephants eating food)


### Experimental Design
<u>Experimental setup:</u> Test how each of the features above affects model accuracy in speech to text models. We will build a mock dataset in order to independently test each of the chosen features. For example, if we were to use a specific language like Kannada, we would likely be testing all 3 of the features at once since Kannada is vastly different from English in all these features. It also allows us to generate ample data needed to train our models. 

<u>Features</u>
1. Silent letters
2. Sound/character omission 
3. Word order

<u>Mock dataset creation:</u>

- nn.Embedding to turn ‘letters’ into their corresponding ‘audio spectrogram’ vector features
- Generate a ‘perfect language’ where all ‘audio spectrograms’ map one to one to ‘letters’, which is the ground truth and control for the project
- Simulate different intricacies of languages (and therefore challenges to the model) by writing python scripts to “mess up” the perfect language
- For example, to simulate English, some particular label n is replaced by two labels k+n with some probability
Some particular label combinations [consonant]+[vowel] is replaced by [vowel]+[consonant]
[vowel] labels get removed entirely
etc.

Architectures to test:
1. RNN
2. Transformers

### Exploring Preprocessing
Finally, as an extension and time permitting, we’d like to explore preprocessing that can be used to improve model accuracy. For example, if we find that models perform poorly when word order becomes non-linear, can we add an ad-hoc algorithm to turn non-linear word order into something that is more “digestible” for the model?

Additionally, current preprocessing includes removing stop words, stemming, removing white spaces, etc.. Perhaps we can generate other rules for different families of languages. This is a section we are still currently thinking about and exploring and would be open to suggestion and feedback. 






