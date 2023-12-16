---
layout: distill
title: Unraveling Social Reasoning in LLMs - A Decision Tree Framework for Error Categorization
description: In this study, we investigate the challenge of social commonsense reasoning in large language models (LLMs), aiming to understand and categorize common errors LLMs make in social commonsense reasoning tasks. Our approach involves expanding upon the preliminary qualitative analyses of social reasoning errors, then developing a decision tree framework for more nuanced and fine-grained error categorization. We will test models such as GPT using this framework. We expect to better understand error types and themes in LLMs' social reasoning, offering insights for improving their performance in understanding complex social interactions.


date: 2023-11-11
htmlwidgets: true

# Anonymize when submitting
# authors:
#   - name: Anonymous

authors:
  - name: Nina Lei
    # url: "https://en.wikipedia.org/wiki/Albert_Einstein"
    affiliations:
      name: Harvard College
  - name: Andrew Zhao
    # url: "https://en.wikipedia.org/wiki/Boris_Podolsky"
    affiliations:
      name: Harvard College
  # - name: Nathan Rosen
  #   url: "https://en.wikipedia.org/wiki/Nathan_Rosen"
  #   affiliations:
  #     name: IAS, Princeton

# must be the exact same name as your blogpost
bibliography: 2022-12-01-distill-example.bib  

# Add a table of contents to your post.
#   - make sure that TOC names match the actual section names
#     for hyperlinks within the post to work correctly.
toc:
  - name: Introduction
  - name: Research Questions
    subsections:
    - name: RQ1
    - name: Experimental Setup
    - name: RQ2
  - name: Methodology
  - name: Expected Outcomes
  - name: References

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


# Unraveling Social Reasoning in LLMs: A Decision Tree Framework for Error Categorization

## Introduction

Despite recent advances and the growth in scale of large language models (LLMs), it's unclear how capable models are of reasoning, especially social commonsense reasoning. <d-cite key="huang_towards_2023"></d-cite>  Tasks involving navigating complex social norms, emotions, and interactions remain a developing frontier in LLM research. 

Prior works like SOCIAL IQA <d-cite key="sap_socialiqa_2019"></d-cite>, ProtoQA, <d-cite key="boratko_protoqa_2020"></d-cite>, Understanding Social Reasoning in Language Models with Language Models <d-cite key="gandhi_understanding_2023"></d-cite> , and SOCIAL CHEMISTRY 101 have provided benchmarking datasets and techniques for social commonsense reasoning and social norms <d-cite key="forbes_social_2021"></d-cite>. Other works, such as Neural Theory-Of-Mind <d-cite key="sap_neural_2023"></d-cite>, explore why models struggle on these datasets and/or try to improve performance, such as by using knowledge graphs. <d-cite key="li_systematic_2022"></d-cite> <d-cite key="chang_incorporating_2020"></d-cite> 

Therefore, our research has two goals: firstly, to expand upon previous research about the types of errors that LLMs make on social reasoning tasks, and secondly, to devise new categories that allow for better granularity when interpreting these mistakes that can help with finetuning models on these errors. 

## **Research Questions**

- RQ1: What are underlying themes in social errors that large language models make?
- RQ2: Are there methods that could potentially address these errors?

## **Methodology**

### **RQ1**

1. **Preliminary Qualitative Analysis:**
    
    We will build upon benchmarking datasets based on Social IQA <d-cite key="sap_socialiqa_2019"></d-cite> and other datasets that provide categorization of social knowledge. An instance of such benchmark is the Social IQA Category dataset <d-cite key="wang_semantic_2021"></d-cite>, which considers four social knowledge categories: Feelings and Characteristics; Interaction; Daily Events; and Knowledge, Norm, and Rules. The authors of this paper found that RoBERTa-large performed the worst in the Feelings and Characteristics category, but did not provide general qualitative or quantitative observations about the types of errors made in each category. We want to better understand these sorts of errors made by the model in these domains.
    
    We plan to conduct an initial qualitative analysis to determine themes in common errors made in each of the four categories. For each model, we plan on sampling 20 or more questions in which the model does not answer correctly, then performing standard qualitative coding procedures to identify a set of common themes in errors for each category.

    Beyond testing the models listed in previous papers, we would like to explore how good GPT-4 is at answering these social commonsense reasoning questions. Given GPT-4's improved capabilities compared to GPT-3, we suspect that this model will perform better; assessing its performance would allow other researchers to draw different insights into how architecture changes and expansions affect social reasoning.

2. **Refinement and Analysis**
    
    Based on the insights gained from the preliminary qualitative analysis, we plan on devising more specific social knowledge categories than the four considered by the Social IQA Category dataset; we aim to construct categories based off of building a decision tree abstraction, where each dimension in the tree corresponds to a trait about the question.
    
    An example set of dimensions for the decision tree abstraction is as follows:
    
    - Dimension 1: Social IQA Category’s four social knowledge categories
    - Dimension 2: Type of question (effects, pre-conditions, stative descriptions, etc.)
        - Note: The authors in the original SocialIQA paper noted that BERT-large found questions related to “effects” to be the easiest and questions about “descriptions” to be the most challenging and claimed that models found stative questions difficult.
    - Dimension 3: Whether reasoning “is about the main agent of the situation versus others.”
        - In Neural Theory-of-Mind? On the Limits of Social Intelligence in Large LMs, the authors argued that models perform much worse when the question is not about the main agent of the situation.
    
    The goal here is to offer a more granular understanding of the categories of errors LLMs make on social reasoning questions. We will then perform another round of Preliminary Qualitative Analysis assessing themes in errors in each category to see if our categories improve on other papers' categories.
    

**Experiment Setup**

- We will qualitatively assign each example under some combination of categories, trying to recreate the purported low performance on these social-reasoning datasets.
- Due to constraints in time, access, and computational resources, we plan on probing models like GPT-3 through through an API, similar to how it was done in the paper Neural Theory-of-Mind? On the Limits of Social Intelligence in Large LMs.
    - Specifically, we will test on GPT-4 to see if these shortcomings still apply.

### **RQ2**

- What we do here is largely dependent upon RQ1 findings; strategies for addressing errors in social commonsense reasoning are largely contingent on the types and themes of errors identified in RQ1.
- We also consider existing approaches to enhancing capabilities of LLMs when it comes to social commonsense reasoning. Namely, in literature, many papers have experimented with the integration of external knowledge bases and fine-tuning models with semantic categorizations of social knowledge.

## Expected Outcomes

- A comprehensive catalog of error types and themes in LLMs concerning social reasoning.
- Insights into the comparative analysis of different models on social reasoning benchmarks.

