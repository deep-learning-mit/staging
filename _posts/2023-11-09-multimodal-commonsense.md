---
layout: distill
title: Multimodal Commonsense Proposal
description: 6.S898 project proposal for analyzing and evaluating the commonsense reasoning performance of multimodal vs text-only models.
date: 2023-11-08
htmlwidgets: true

# Anonymize when submitting
# authors:
#   - name: Anonymous

authors:
  - name: Vincent Lin

# must be the exact same name as your blogpost
bibliography: 2023-11-09-multimodal-commonsense.bib

# Add a table of contents to your post.
#   - make sure that TOC names match the actual section names
#     for hyperlinks within the post to work correctly.
toc:
  - name: Background
  - name: Related Work
  - name: Implementation & Evaluation

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

In recent years, language models have been proven to be quite proficient in producing human-like text, computing somewhat semantically-meaningful and human-interpretable word and token embeddings, and generating realistic conversation. However, there is a vast distinction between mimicking human linguistics from data and forming an understanding of the world and its abstract connections from data. The latter describes the commonsense knowledge of a language model, or its ability to reason about simple relationships, interactions, and general logic of the world.

Previous work has been completed evaluating the commonsense capabilities of langauge models, and with the vast sizes of LMs and the even vaster data availble today, language models' commonsense performance has grown increasingly close to human performance -- but not quite <d-cite key="li2021"></d-cite>. From textual data alone, models still perform worse than humans with a significant margin of error. Yet, humans don't learn to reason about the world from text alone; many, many different modes of perception contribute to our knowledge of reality. Can we imbue deep learning models with other modes of input to similarly augment their reasoning skills?

In this project, I propose an investigation and evaluation of multimodal deep learning models for commonsense reasoning. When compared to standard language models, multimodal models have a more diverse set of input/training data that, perhaps, grants them a richer representation of the data. For example, vision-text models can be trained on the same textual data as language models, but the association of images and visualized objects with text embeddings provides a more comprehensive "understanding" of the objects and their interactions with their environment. Do different types of auxiliary inputs types provide multimodal models with any additional commonsense information? In the context of model representations and embeddings, how do the multimodal representations differ from those of the (text-only) unimodal? How are they similar? When observing the relationships between embeddings within the multimodal model (e.g., latent-space distances), does the multimodal affect the relative similarity between words/objects? Do these augmented relationships benefit multimodal models in commonsense reasoning at all?

## Related Work

Several works have evaluated the commonsense capabilities of unimodal language models. Li et al., 2021 <d-cite key="li2021"></d-cite> analyzes the performance of the Gopher language model in zero-shot and few-shot learning with varying model sizes. They find that their LM performed relatively well in physical commonsense (explained further below), but worse in social commonsense. Zhao et al., 2023 <d-cite key="zhao2023"></d-cite> measure large language models' commonsense performance in the context of simple task planning, e.g., in robotics, observing that performance varies depending on the particular task and the length of the descrption for the task. Saharia et al., 2022 <d-cite key="saharia2022"></d-cite> propose a text-to-image multimodal model and evaluate the depth of its text language understanding.

## Implementation & Evaluation

For this project, I will choose to focus on vision-text models to evaluate multimodal performance. It's important to note that different types of commonsense exist, and vision-text models may, intuitively, perform better at physical commonsense tasks than, say, social tasks, which will be a crucial distinction in evaluation. Reliable and relatively compact language models already exist with pretrained weights and relatively solid performance in general NLP tasks (e.g., transformer models from Hugging Face <d-cite key="huggingface"></d-cite>), so I will plan to use these as reference. I may choose to implement more of the vision-text model from scratch (though carefully, so as not to have lackluster text processing in the multimodal model impact any comparison with the reference LM). However, if complications do arise, preimplemented multimodal models may also be used for reference <d-cite key="saharia2022"></d-cite>.

Many benchmarks are available for evaluating the commonsense capabilities of language models. I will focus on multiple choice evaluation, where given a short story or background prompt, a model must choose the most reasonable answer or continuation. Multiple choice benchmarks provide a more concrete and reliable metric for determining similarity to “human” judgement. A brief summary of some potential benchmarks is given below:

__HellaSwag__<d-cite key="zellers2019"></d-cite>: Designed to evaluate physical, grounded, and temporal common sense. Given a short description/prompt, the model must choose the correct continuation from four choices. The "stories" are produced from video captions or other passages.

{% include figure.html path="assets/img/2023-11-09-multimodal-commonsense/hellaswag.png" class="img-fluid" %}

__Social IQa__<d-cite key="sap2019"></d-cite>: Evaluates a model's social common sense. This dataset is comprised of social situations of interactions between people, evaluating a model's knowledge of emotion, mental states, etc.

{% include figure.html path="assets/img/2023-11-09-multimodal-commonsense/socialiqa.png" class="img-fluid" %}

__PIQA__<d-cite key="bisk2019"></d-cite>: Another physical common sense benchmark, where given a short question or situational prompt, models must select a solution between two options. PIQA focuses on physical interaction.

{% include figure.html path="assets/img/2023-11-09-multimodal-commonsense/piqa.png" class="img-fluid" %}