---
layout: distill
title: Leveraging Representation Engineering to Evaluate LLM’s Situational Awareness
description: We present a method to tell whether LLMs are drawing from knowledge not explicitly mentioned in the prompt by examining token-level representations.
date: 2023-11-08
htmlwidgets: true

# Anonymize when submitting
# authors:
#   - name: Anonymous

authors:
  - name: Alex Hu
    url: "https://www.linkedin.com/in/alexander-hu/"
    affiliations:
      name: MIT
  - name: Carl Guo
    url: "https://www.carlguo.com/"
    affiliations:
      name: MIT

# must be the exact same name as your blogpost
bibliography: 2023-11-08-representationengineering-incontextlearning.bib  

# Add a table of contents to your post.
#   - make sure that TOC names match the actual section names
#     for hyperlinks within the post to work correctly.
toc:
  - name: Background
  - name: Motivation
  - name: Sources


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
Emerging capabilities in deep neural networks are not well understood, one of which is the concept of “situational awareness,” an emergent LLM capability where they understand whether they are in training, testing, or deployment. This behavior can emerge from the fact that training datasets contain articles about LLMs, AI model training, testing, and deployment. If an LLM were to possess “situation awareness,” it might give misleading results on safety tests before deployment or deceptively align to human feedback in fine-tuning. Understanding and evaluating LLM’s capability of “situational awareness” can increase LLM’s safety and usefulness. 

Because “situational awareness” is a loaded concept,<d-cite key = "berglund2023measuring"></d-cite> [Berglund et al. (2023)](https://arxiv.org/pdf/2309.00667.pdf) study a proxy capability that they coin “sophisticated out-of-context reasoning” (SOC), where LLMs utilize data from pre-training/fine-tuning corpora during inference on an unrelated piece of text without specifically being prompted to do so. Specifically, they finetune LLMs to mimic a chatbot to, say, answer the questions in German by only giving them the description that it speaks only German but not German text. Here, the model is evaluated on a task where it needs to perform much more sophisticated reasoning than direct retrieval from the training set.

Another inspiring field of work is to understand and interpret the mechanistic internals of deep learning models. One such inspiring work is Zou et al. (2023)’s paper on Representation Engineering (RepE), where they construct a set of training text stimuli to elicit LLM’s beliefs, split them into pairs, and use PCA to find a reading vector to transform the model representation then when given new tests. This approach allows us to elicit readings of representation and control such representation. Similarly, [Meng et al. (2023)](https://arxiv.org/pdf/2210.07229.pdf) present ways to edit memory in Transformers about certain representations. 
## Motivation
[Berglund et al. (2023)](https://arxiv.org/pdf/2309.00667.pdf)’s work is limited in the sense that it studies out-of-context learning in toy settings after fine-tuning the model on task descriptions. Instead, we aim to discover token-level representations indicating how much models perform sophisticated out-of-context reasoning in more realistic test settings leveraging representation control and editing tools mentioned above. These tools allow us to construct artificial examples of out-of-context learning while maintaining the overall model performance on other tasks, making the evaluation more realistic. Finally, we construct features from the final layer of our LLM when performing inference and group these features depending on whether the generated token relies or does not rely on out-of-context reasoning. We’ll use the Representation Reading methods presented in [Zou et al. (2023)](https://arxiv.org/pdf/2310.01405.pdf) to review the context where the model attends to and discover directions that indicate such reasoning (or the lack thereof), and compare our findings against the fine-tuning approach.

