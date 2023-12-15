---
layout: distill
title: 6.S898 Project Proposal
description: Increasing Context Length For Transformers
date: 2023-11-08
htmlwidgets: true

# Anonymize when submitting
# authors:
#   - name: Anonymous

authors:
  - name: Annie Wang
    affiliations:
      name: MIT

# must be the exact same name as your blogpost
bibliography: 2023-11-08-increasing-context-length-for-transformers.bib  

# Add a table of contents to your post.
#   - make sure that TOC names match the actual section names
#     for hyperlinks within the post to work correctly.
toc:
  - name: Overview

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
Modern-day transformers often aim to solve problems that utilize large inputs with long-range dependencies. For instance, the development of sophisticated LLMs such as GPT-4 has given rise to text prompts that are several hundred words in length, where the first sentence may impact the interpretation of the last. Today’s transformers—particularly LLMs—often come with a maximum context length, so that excessively long inputs are not accepted. Yet, this context length is often exceeded when trying to solve complex problems within the model’s domain; as an example, we consider the task of summarizing long documents via GPT-4.

Evidently, a transformer’s maximum context length greatly affects the types of information it can process and the questions it can answer; larger context lengths would allow transformers to solve even more complex problems.

However, the time complexity of transformers is quadratic with regard to the input length. In the traditional transformer model, each element in the input sequence is mapped to one or more tokens, and each token attends to every token prior to it—making the attention mechanism a relatively expensive computation. As a result, strategies for decreasing the context length of large inputs is a very relevant topic in the development of transformers.

In this project, we investigate the effects of large context length on transformers, along with current methods of increasing context length. Additionally, we evaluate the advantages and disadvantages of current approaches for increasing context length and attempt to apply them to different transformer-based problems. Finally, we propose a new scheme for increasing context length. We test this scheme via ablation studies and aim to explain why or why not it does not perform as well as current approaches.

A more detailed breakdown of the project plan is provided below.

| Task      | Description |
| --------- | ----------- |
| Investigate effects of increasing context length without limiting the number of tokens that must be attended upon. | Train transformers to solve the same problem (e.g., language generation based on a provided dataset), but with different maximum context lengths. Assess the performance of the resulting models, including how well they are able to solve the initial problem and how long they take to train and generate data. |
| Survey current approaches for increasing context length. | Investigate how current approaches aim to increase context length while reducing overall time complexity. Discuss different advantages and disadvantages of current methods. |
| Assess advantages and disadvantages of current approaches for increasing context length, as applied to specific transformer-based problems. | Investigate whether certain methods for reducing context length work better for certain problems than for others. Why or why not? Investigate whether proposed methods work as well in practice as they do in theory. |
| Investigate a new scheme for increasing context length. | Using existing knowledge, propose a new scheme for increasing context length and provide an explanation as to why the selected scheme was chosen. |
| Test the proposed scheme for increasing context length. | Attempt to solve an existing transformer-based problem using the new scheme. Compare results to results using existing approaches. Provide a hypothesis as to why the new scheme works or does not work. |
