---
layout: distill
title: Multilingual Representations in Embeddings Models [proposal]
description: Learning how encoder-only transformers represent language, and testing if you can teach an old model to speak a new language.
date: 2023-11-09
htmlwidgets: true

# Anonymize when submitting
# authors:
#   - name: Anonymous

authors:
  - name: Spruce Campbell
    url: "spruce.world"
    affiliations:
      name: MIT, CSAIL
  - name: Will Hathaway
    url: "willhath.com"
    affiliations:
      name: MIT, CSAIL

# must be the exact same name as your blogpost
bibliography: 2023-11-09-multilingual-representations-in-embeddings-models.bib

# Add a table of contents to your post.
#   - make sure that TOC names match the actual section names
#     for hyperlinks within the post to work correctly.
toc:
  - name: Motivation
  - name: Method

# Below is an example of injecting additional post-specific styles.
# This is used in the 'Layouts' section of this post.
# If you use this post as a template, delete this _styles block.
_styles: >
  // insert CSS here
---

## Motivation

Recently, [embeddings models](https://platform.openai.com/docs/guides/embeddings) have become incredibly popular as LLMs become integrated into tools and applications. Embeddings models (specifically, Siamese encoder-only bidirectional Transformers) are the state-of-the-art method in an old problem in computer science, the retrieval problem. Embeddings are often used in settings like recommendation algorithms, similarity search, and clustering, and have recently found use in Retrieval-Augmented Generation, where an LLM is given context that embedded nearby in order to answer questions more truthfully. However, many of the most high-performance embeddings models are trained on English only, which means they suffer greatly at applications in other languages, and are inaccessible to most of the world. In this blog post, we summarize the history of embeddings, detail the training regime of a modern embeddings model, train both monolingual and multilingual models, and investigate whether it is possible to fine-tune in multilingual capability to a pretrained monolingual model.

## Methods

Training a modern embeddings model can be broken down into three parts:

### 1. Pretraining

It is valuable to start with a language model that has already learned some inner representation of language. This makes the embeddings task significantly easier, since the model must only learn to condense this inner representation into a single high-dimensional dense vector space. Large language models have shown incredibly good adaptability and generalization, so the natural choice as an architecture is a Transformer. GPT-style models are extremely popular for text generation, but are not a good choice for embeddings because they have a _causal attention mask_, and so cannot move information backwards from a sequence. Embeddings models are always given the full text string, however, and so there is no need to arbitrarily throw out this information. BERT-style models are a much more common choice. BERT (Bidirectional Encoder Representations from Transformers), was one of the first LLMs to see real-world success, and is highly fine-tunable. The BERT architecture looks like this:

{% include figure.html path="assets/img/2023-11-09-multilingual-representations-in-embeddings-models/bert.png" class="img-fluid" %}

_(continue to discuss BERT loss objectives, pooling, and constructing embeddings)_

### 2. Training

_(Discuss how BERT can be paired with itself to create a pair encoder, that can then be used to construct a contrastive loss, resulting in an embeddings space)_

### 3. Finetuning

_(Discuss how many real-world use cases depend on queries being embedded similar to text, and how finetuning is performed on this. Discuss the use of hard negatives rather than in-batch negatives.)_

The question this blog post aims to explore is whether it is possible to insert new languages at the training or finetuning stage, using publicly available datasets of text pairs. If successful, it would mean that the encoder learnt some map from one language onto the embedding space of another. This implies that it is possible to approximate translation, at a conceptual level, with a linear transformation. We will study the results on various language pairs, and compare to a pretrained multilingual model.

## Notes on Methods (proposal only)

There is a good candidate pair of models for this blog post: [e5-base](https://huggingface.co/intfloat/e5-base-v2) and [multilingual-e5-base](https://huggingface.co/intfloat/multilingual-e5-base) share the same architecture and were trained by the same team, showing roughly equivalent performance modulo the expected multilingual hit to English performance. Graciously, the code is also public at [unilm/e5](https://github.com/microsoft/unilm/tree/master/e5). Datasets of multilingual text pairs include [CCMatrix](https://ai.meta.com/blog/ccmatrix-a-billion-scale-bitext-data-set-for-training-translation-models/) and [NLLB](https://huggingface.co/datasets/allenai/nllb).

However, we are also in the midst of pretraining our own model with significantly better performance than e5, and a longer context length. This is achieved through lots of hacks, as well as new training objective. We hope this model will be ready to use in the blog post.

Compute is not a concern. We have access to 7,500 A100-hours (>1000 exaFLOP) for this project, thanks to a compute grant from Stability AI.