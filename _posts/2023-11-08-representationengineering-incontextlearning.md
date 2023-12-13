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

# Introduction 
Emerging capabilities in deep neural networks are not well understood, one of which is the concept of "in-context learning" (ICL), a phenomenon where the a Large Language Model (LLM)'s understanding of the prompt and ability to answer accordingly drastically increases after being shown some examples that answer the question. Evaluating in-context learning and understanding why the behavior happens is both an interesting theoretical research question and a practical question that informs directions to conduct research that further advances LLM capabilities by, say, exploiting more of in-context learning. 

We attempt to explore the phenomenon of in-context learning by leveraging another exciting field of work on mechanistic interpretability where researchers set out to understand model behaviors by interpreting and editing internal weights in models. One such work that we base on is Representation Engineering by Zou et al. (2023)<d-cite key="zou2023representation"></d-cite> , where they construct a set of training text stimuli to probe LLM activations and use such stimuli to identify a direction that accurately predicts the underlying concept based on the neural activations of the model. This approach allows us to elicit readings of representation and control such representation.

We propose to use methods in Zou et al. (2023) <d-cite key="zou2023representation"></d-cite> to evaluate in-context learning by constructing artificial examples of in-context learning on binary classication tasks. We find a reading vector that shows high neural activity after the model is stimulated with the context pairs; such a "Context Vector" indicates the context the models draws from. We then explore the results of controlling the activations along the "Context Vector" direction, in the hope that editing the activitions would further boost the performance on top of in-context learning. We compare the model outputs on the classification datasets in a zero-shot setting and a setting of natural in-context learning, with the "Context Vector" amplified, and suppressed. We found that such model weight editing accomplish XXX. 

While we find boosting performance through such editing to be challenging and sometimes finicky to tune, we find the results to be promising on editing weights to suppress the context that the model draws from and drastically reducing the performance. We hope that this work can serve as a stepping stone to further understand the phenomenon of in-context learning and how to leverage it to improve LLMs.
 
## Background & Related Work

### In-Context Learning (ICL)
An LLM is frequently aseked to perform a task in inference time that many realized providing some examples of how to answer the task can drastically improve the model's performance. This phenomenon is called in-context learning. 

For example, Zhou et al. (2022) <d-cite key = "zhou2022teaching"></d-cite> evaluates how LLM can become better at solving algorithmic problems through in-context learning, a task that LLM traditionally struggles at. 

In other scenarios, the LLM does not need to rely on prompts at all and can deduce the pattern from the few-shot examples alone to predict the answer. While there is no universal definition of in-context learning and its meaning has shifted over time, we define it as the performance boost to answer questions based on a limited amount of examples (as the context). 


The concept is first popularized by the GPT-3 paper (Brown et al. (2020) <d-cite key = "brown2020language"></d-cite>), where the authors observe that models can make “increasingly efficient use of in-context information” by "developing a broad set of skills and pattern recognition abilities" during pretraining and then "rapidly adapt to or recognize the desired task" during inference. The concept puzzles many researchers because LLMs are not explicitly pre-trained to perform such tasks, but rather on next-token prediction, and the model's ability to perform such tasks is not well understood.

Min et al. (2022) <d-cite key = "min2022rethinking"></d-cite> observes that such ICL phenonemon is observed as long as examples are given, and a mismatch between input and output pairs would not hinder the ability of models performing ICL and thus its performance on the tasks. Wei et al. (2023) <d-cite key="wei2023larger"></d-cite> further corrobates this work by finding on small models but show that as models scale, the ability to pick up on flipped patterns when given in-context examples with flipped labels and override semantic priors is stronger.

### Theories on why ICL happens
While the concept of ICL is well studied, the underlying mechanism of ICL is not well understood. Xie et al. (2022) <d-cite key = "xie2022explanation"></d-cite> explains the phenomenon of ICL as an Implicit Bayesian Inference, where the in-context learning prompt serves as a stimulus for the model to go "locate" corresponding concept stored in the model's latent space that the LM has learned implicitly during pre-training. They study this by generating a simple pretraining distribution that parameterizes the transition of a Hidden Markov Model (HMM) and another prompting distribution. In this setting, the authors reduce the ICL task to Bayesian inference when asked about prompts from another distribution and map it through the HMM onto the pretraining distribution. 

Akyürek et al. (2022) <d-cite key = "akyürek2023learning"></d-cite> further explains that Transformer-based in-context learners implement standard learning algorithms implicitly. von Oswald et al. (2023) <d-cite key="vonoswald2023transformers" ></d-cite>claims that Transformer-based in-context learners is similar to gradient-based meta-learning formulations where they found that the Transformer can learn smaller models of a certain concept by gradient descent in their forward pass.

Furthermore, Olsson et al. (2022)  <d-cite key = "olsson2022context"></d-cite> draws parallel from ICL to a more understood phenomenon of Induction Head, where attention-only Transformers picks up on the algorithm to predict next tokens by searching for a previous occurance of the last token and copying the same next token from previous occurences. They claim that this can be a potential mechanism to explain ICL. 

### Model Editing & Representation Engineering

We’ll use the Representation reading and controls methods presented in [Zou et al. (2023)](https://arxiv.org/pdf/2310.01405.pdf) to understand the context where the model attends to and discover directions that indicate such reasoning. 

Relatedly, there have been a recent surge in research related to model knowledge editing, including Meng et al. (2023) <d-cite key = "meng2023massediting"></d-cite>, Zhong et al. (2023) <d-cite key = "zhong2023mquake"></d-cite>, and Hernandez et al. (2023) <d-cite key = "hernandez2023inspecting"></d-cite> that demonstrate different methods for locating and editing factual associations. Other work, including Shao et al. (2023) <d-cite key="shao2023gold"></d-cite> and Belrose et al. (2023) <d-cite key="belrose2023leace"></d-cite>, have shown results on concept erasures. 

Li et al. (2023) <d-cite key="li2023inferencetime"></d-cite> proposes Inference-Time Intervention, where they find directions of causal influence on "truthfulness" data and increase the activations along that direction to increase truthfulness, scoring better on the TruthfulQA dataset. 

# Experiment Setup

## Datasets

We adopt the 26 dataset used by Min et al. (2022) <d-cite key="min2022rethinking"></d-cite> plus 4 extra datasets, including `rotten_tomatoes`, `ade_corpus_v2-classification`, `tweet_eval-irony`, `tweet_eval-stance_climate` to evaluate in-context learning. Following Min et al. (2022)<d-cite key="min2022rethinking"></d-cite>, we only use the test set to avoid potential cross-contamination with the data that the model is pretrained on.  reserve `k=64` examples in the test for few-shot training, and the rest are used for testing. 

### Training Data Generation 

For training, we construct a set of context pairs for each dataset, each context pairs containing the same examples but different instructions. The instructions are "Pay attention to the following examples" and "Ignore the following examples" respectively, in the hope that by stimulating two opposites and examining the difference, we can find a Context Vector that represents what the model draws from.

A sample training data input using the `rotten_tomatoes` dataset is as follows: 


> [INST] Pay attention to the following examples: [/INST] 

> offers that rare combination of entertainment and education.

> positive.

> a sentimental mess that never rings true .

> negative.

> [INST] Ignore the following examples: [/INST]

> offers that rare combination of entertainment and education.

> positive.

> a sentimental mess that never rings true .

> negative.

Each context pair is identical except for the instructions. We use the context pairs to stimulate the model to learn the context and use the context vector to control the model's behavior.

### Testing Data Generation

For testing data, we use 3 input-labels pairs as the prompt, with the first two pairs serving as the in-context examples, and the last pair serving as the question that we actually want to test on, obfuscating the label from the prompt. 

A sample testing data input using the `rotten_tomatoes` dataset is as follows:

Input: 
> [INST] offers that rare combination of entertainment and education. [/INST]

> positive.

> [INST] a sentimental mess that never rings true . [/INST]

> negative.

> an odd , haphazard , and inconsequential romantic comedy .

Label:
> negative.

## Model

We have explored using two models with 7 billion parameters, including `Mistral-7B-Instruct-v0.` and `Llama-2-7b-hf`; while we have found preliminary results consistent between the two models, all of our results later reported are from `Mistral-7B-Instruct-v0` for consistency and due to a constraint on computational power and time. 

## Training Infrastructure

We used the MIT Supercloud infrastructure and a local machine with a single RTX 4090 GPU to train the model.

# Results 

We present results first on finding the Context Vector in the embedding space, then on using the Context Vector to control model outputs and evaluate their performance.

## Representation Reading

We use 

## Representation Control



