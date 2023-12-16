---
layout: distill
title: Project Proposal
description: Using Adversarial Images to Jailbreak Large Visual Language Models
date: 2023-11-9
htmlwidgets: true

# Anonymize when submitting
# authors:
#   - name: Anonymous

authors:
  - name: Julie Steele
    url: "mailto:jssteele@mit.edu"
    affiliations:
      name: MIT

  - name: Spencer Yandrofski
    url: "mailto:spencery@mit.edu"
    affiliations:
      name: MIT

# must be the exact same name as your blogpost
bibliography: 2023-11-9-adversarial-image-jailbreak.bib  

# Add a table of contents to your post.
#   - make sure that TOC names match the actual section names
#     for hyperlinks within the post to work correctly.
toc:
  - name: Proposal
  # - name: Images and Figures
  #   subsections:
  #   - name: Interactive Figures
  - name: Citations
  - name: Footnotes
  - name: Code Blocks
  - name: Layouts
  - name: Other Typography?

--- 

## Proposal : Using Adversarial Images to Jailbreak Large Visual Language Models
We hope to study using adversarially crafted images as inputs to large visual language models (like gpt4, where one can input an image) to jailbreak the language model. Jailbreaking entails bypassing alignment efforts for the model not to speak on dangerous/mean topics. Creating adversarial images to trick image classifiers has been widely studied, and methods including fast gradient sign method, Carlini-Wagner’s L2 attack, Biggio's attack, Szegedy′s attack, and more (see https://arxiv.org/pdf/1711.00117.pdf, https://link.springer.com/article/10.1007/s11633-019-1211-x) have been effective. There have also been successful efforts in optimizing token inputs to jailbreak language models. The recent creation of visual language models allows for an oportunity to combine adversarial images and jailbreaking.  

We will investigate the applicability of each of these attacks for visual language models, and then compare a couple of them on effectiveness at jailbreaking the model. Will some work unexpectedly better/worse compared to image classification adversarial attacks? Why? We would start with trying white-box attacks (viewing the weights of the visual language model). One question we will have to tackle is what is a good measure of jailbreaking success we have (as opposed to classification accuricary), and if we can find an objective measure to use in the model. We would use pretrained open source MiniGPT4 for the experiments. 

All parts of this project are very subject to change, and we would love ideas and mentorship from course staff! 

## Other Ideas
Training a GAN: model 1 makes adversarial images, model 2 finds the fake 
Jailbreaking an LLM, experimenting over different levels to do the optimization (tokens? post-embedding?)
Adversarial images for jailbreaking language models (see https://arxiv.org/abs/2306.13213): This paper compares text attacks for jailbreaking and image attacks. Since images are differentiable, they work better. Adversarial training and robustness certification are two methods to try to fix this, but likely not to prevent image attacks. 

## Related Work 

* https://arxiv.org/abs/2306.13213 **extremely related, building off of 
* https://arxiv.org/pdf/1711.00117.pdf 
* https://arxiv.org/pdf/2002.02196.pdf 
* https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1184/reports/6906148.pdf  
* https://mpost.io/openai-develops-jailbreak-gan-to-neutralize-prompt-hackers-rumors-says/  
* https://arxiv.org/abs/2307.15043 
* https://ieeexplore.ieee.org/abstract/document/7727230?casa_token=82pyRsetYb0AAAAA:GsItW94vrH-aqxxl8W365qG_CBDt_lSyMfCn33bD32HNonSt2LKd_0QZLve7rnrg9fmeLmqYsw 
* https://link.springer.com/article/10.1007/s11633-019-1211-x 

