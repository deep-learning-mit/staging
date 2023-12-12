---
layout: distill
title: Denoising EMG signals
description: The future of brain-computer interfaces rests on our ability to decode neural signals. Here we attempt to ensemble ML techniques to extract useful information from sEMG signals to improve downstream task performance.
date: 2023-11-08
htmlwidgets: true

# Anonymize when submitting
authors:
  - name: Anonymous

# authors:
#   - name: Prince Patel
#     url: "https://ppatel22.github.io/"
#     affiliations:
#       name: MIT


# must be the exact same name as your blogpost
bibliography: 2023-11-08-denoising-EMG-signals.bib  

# Add a table of contents to your post.
#   - make sure that TOC names match the actual section names
#     for hyperlinks within the post to work correctly.
toc:
  - name: Project Proposal
    subsections:
    - name: Introduction
    - name: Literature Review
    - name: Project Proposal
    - name: Methodology
    - name: Anticipatede Impact
  # - name: Equations
  # - name: Images and Figures
  #   subsections:
  #   - name: Interactive Figures
  # - name: Citations
  # - name: Footnotes
  # - name: Code Blocks
  # - name: Layouts
  # - name: Other Typography?

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
## Project Proposal
### Introduction
Brain-machine interfaces (BCIs) have the potential to revolutionize human-computer interaction by decoding neural signals for real-time control of external devices. However, the current state of BCI technology is constrained by challenges, particularly the high signal-to-noise ratio (SNR) in nerve recordings, limiting widespread adoption beyond clinical settings. To address this, significant advancements in both hardware and software have been pursued, focusing on enhancing the measurement and decoding of neural signals.

### Literature Review
Recent innovations have attempted to mitigate SNR challenges using software-based techniques, such as employing preprocessing methods like low/high-pass filters, Fourier transforms, and outlier removal. Notably, the introduction of BrainBERT <d-cite key="BrainBERT"></d-cite> presented a transformative approach with a transformer model designed to extract richer representations of neural signals, primarily for gesture recognition tasks. While promising, limitations exist, including the use of intracranial recordings, limited dataset size, and minimal validation on downstream tasks, underscoring the need for further exploration.

### Project Proposal
In this research, I aim to develop and train a denoising auto-encoder empowered with self-attention mechanisms tailored to preprocess surface electromyography (sEMG) recordings efficiently. Leveraging a substantial dataset <d-cite key="sEMGdataset"></d-cite> comprising diverse sEMG recordings, encompassing raw and preprocessed signals related to various finger movements, I plan to design the autoencoder to optimize the reconstruction loss between the preprocessed recordings and their corresponding reconstructions, departing from the conventional approach of raw signal reconstruction.

### Methodology
Drawing inspiration from the transformer architecture, notably BrainBERT, I will adapt the encoder module to effectively capture intricate temporal dependencies within the EMG signals. Through strategic modifications and enhancements to the model, I aim to bolster the learned encodings' performance in downstream tasks, emphasizing gesture recognition and potentially extending to other relevant applications.

### Anticipated Impact
The proposed study anticipates fostering a novel framework for preprocessing EMG signals, contributing to the advancement of practical BCI applications outside clinical environments. By addressing SNR challenges and enriching the learned representations through a sophisticated denoising auto-encoder with self-attention, this research holds promise for accelerating the development and adoption of robust, noninvasive BCI solutions for diverse real-world contexts.


