---
layout: distill
title: New Synthesis Approach for Personalized LLMS
description: 
date: 2023-12-12
htmlwidgets: true

# Anonymize when submitting
# authors:
#   - name: Anonymous

authors:
  - name: Daniel Papacica
    url: 
    affiliations:
      name: MIT
  - name: Ben Ebanks
    url: 
    affiliations:
      name: MIT

# must be the exact same name as your blogpost
bibliography: 2023-11-09-PersonalizedGeneration_w_LLMAgents.bib  

# Add a table of contents to your post.
#   - make sure that TOC names match the actual section names
#     for hyperlinks within the post to work correctly.
toc:
  - name : Introduction 
  - name: Literature Review
  - name: Description of methods & experiments
    subsections:
    - name: The Baseline Implementation
    - name: Overview of Modification & Experiments
    - name: Experiment / Word2Vec vs GloVe
  - name: Analysis / Evaluation of Results
  - name: Results
  - name: Conclusion / Discussion of Limitations
    subsections:
    - name: Limitations
    - name: Next Steps
  - name: Bibliography

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

Deep learning has revolutionized the way in which humans interact with the world around them. Its growing ability to ingest vast amounts of data, automate feature extraction, and learn complex patterns and nuances among inputs have contributed to breakthroughs in healthcare, natural language processing, computer vision, and more. A particularly exciting avenue of this innovation has been in the burgeoning field of personalized text generation, which aims to produce text that resembles the style, tone, and word choice taken on by a particular user. Significant advancement in this field has the potential to create more effective forms of communication for individuals with disabilities, personalize educational content, and enhance user interactions with chatbots and virtual assistants, all contributing to a better overall user experience. 

In an effort to make the availability of personalized text generation more wide-scale, researchers have conducted several studies in the field, centering their approach to the generation of domain-specific personalized text (utilizing domain-specific features/knowledge). Notable studies conducted include [Towards Controllable and Personalized Review Generation](https://arxiv.org/pdf/1910.03506.pdf), which utilizes a product description and self-attentive recursive autoencoders to generate a personalized review [[1]](#1), [Knowledge-Enhanced Personalized Review Generation with Capsule Graph Neural Network](https://arxiv.org/pdf/2010.01480.pdf), which constructs a model based on a CapsGNN, and [Research on user granularity-level personalized social text generation technology](https://iopscience.iop.org/article/10.1088/1742-6596/2294/1/012015/pdf), which utilizes an encoder and decoder for text generation [[2]](#2). A lesser explored part of the field and an area that we have chosen to explore for our final project is embedding in the ability to generate personalized text across domains without domain-specific features [[3]](#3). Our project draws inspiration from ["Teach LLMs to Personalize – An Approach inspired by Writing Education”](https://arxiv.org/pdf/2308.07968.pdf), which includes a promising multi-step framework that retrieves, summarizes, ranks, and synthesizes a user’s past documents to generate a personalized version of the document at hand [[4]](#4). 

A critical aspect of the workflow discussed in the LLM personalization paper and an area that we believe can be improved upon using some of the methods discussed in 6.S898 this semester is the way in which the model synthesizes past documents. Throughout the paper, we will be exploring two creative approaches to synthesis that utilize vector word embeddings to pull relevant words from past documents in an effort to improve the models ability to personalize text.


# Literature Review
An integral part of our exploration project was experimenting with using less data and smaller models to see how performance degrades with respect to the approach discussed in the personalization for LLMs paper (no open source code attached as the project is currently being worked on by researchers at Google). Experimentation required taking an extensive look at the steps involved in the original implementation, gaining an in-depth understanding of the deep learning principles discussed, and optimizing training and compute under machine constraints to process vast amounts of real-world data.

The problem formulation for the approach to personalized text generation discussed in the paper can be stated as the following: Given the immediate context of a current document (first k characters) written by a user and access to their past documents, can we develop a model that generates text that is similar to the text of the current document (similarity evaluated by calculating Rouge-1, Rouge-2, Rouge-L, and Bleu scores) . As mentioned earlier, the framework for answering this problem formulation involves first obtaining outputs for retrieval, ranking, summarization, and synthesis, and then feeding these distinct parts into an LLM to produce a personalized body of text (we ignore the auxiliary task of training the LLM to distinguish the owners of written documents for the purposes of this project). 

The retrieval discussed in the paper uses two methods of outputting relevant documents: sparse retrieval, which compares past documents to the current context using the popular BM25 ranking algorithm, and dense retrieval, which uses a transformer-based text-to-text model to map and compare documents in a 768 dimensional vector space. The ranking step then takes this input, orders documents based on their BM25 scores or cosine similarity when compared with the immediate context, and truncates the input to 2500 characters to only take the top documents. The summarization step then summarizes the top ranked past documents in two ways: context independent summarization, which finetunes an LLM on publicly available data and applies this model to the top ranked entries, and context dependent summarization, which uses weak labels (generated from immediate context) to generate a summary in line with the contents of the current document. A visualization of the approach to the structure can be seen below.

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/2023-11-09-PersonalizedGeneration_w_LLMAgents/gen_structure_overview.png" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    An overview of the infrastructure used to process documents and fine tune the personalized generative model.
</div>

The paper currently explores two methodologies for synthesis: (1) context dependent synthesis, which simply uses the top 20 frequently used keywords from a user’s past documents and (2) context dependent synthesis, which uses weak labels to find text from past documents similar to the immediate context of the document to be personalized. After carefully analyzing the two methodologies, we found that by focusing on keywords the synthesized text is missing an in-depth understanding of sentence structure and semantics that are crucial to personalization.

To enhance this step of the text generation process, we have explored several new methods of synthesis and have landed on two approaches with one utilizing the Word2Vec model and the other using GloVe. We have chosen these methods because they both use unique embedding space attributes to form important relationships between texts. Both networks use the method of creating a centroid of the current document that exists in vector space and output words from top ranked past documents that exist close to this centroid. By doing this, we are essentially selecting words (after filtering out synonyms and stopwords) that are in line with the theme of the current document, which will provide the LLM with more thematically relevant synthesized entries that should in theory generate a more personalized output. 

As an additional research consideration, we explored the effect of passing in the output from both the context independent synthesis discussed in the paper and our auxiliary method of using Word2Vec or GloVe compared to passing in just one of the methods of synthesis. The motivation for doing so came from our initial hypothesis that the combination of both methods of synthesis would enable the LLM to learn complex interactions between important words (results from context independent synthesis) and thematic words (GloVe/Word2Vec) that could lead to better personalization of the final output. A more detailed explanation of the implementations of our proposed approaches will be shown in the following section.


# Description of methods & experiments

## The Baseline Implementation
Our methodological approach began by re-implementing the baseline model from the "Teach LLMs to Personalize" paper. We utilized two datasets mentioned in the research paper: CNN_DailyMail ([CNN_DailyMail](https://huggingface.co/datasets/cnn_dailymail))and Amazon Review Data for Books ([Amazon_review_data](https://cseweb.ucsd.edu/~jmcauley/datasets/amazon_v2/)). To enhance efficiency of compute time, we streamlined the data by reducing its size, ensuring a quicker fine-tuning process while retaining data integrity. We also utilized the [T5-base model](https://huggingface.co/t5-base), a smaller model than the T5-11b model mentioned in the paper, for summarization and the personalized generation model. Furthermore, we opted to use the context-independent methods for both summarization and synthesis because the research paper results indicated that their effectiveness is closely comparable to the context-dependent methods. For fine-tuning the summarization model, we utilized a 10 percent subset of the CNN daily mail dataset (311k datapoint original size) with the AdamW optimizer (seeing AdamW is a comparable optimizer to Adafactor, which is what was used in the "Teach LLMs to Personalize" paper), ensuring a balance between efficiency of tuning and comprehensive learning. This set the foundation for our exploration of advanced text synthesis techniques by giving us a base fine tuning and data processing infrastructure. On top of this, the changes we made to the amount of data used along with utilizing a smaller T5 model allowed us to analyze whether the final evaluation results degraded significantly when making the infrastructure of fine tuning the personalized generation model more compact.

## Overview of Modification & Experiments
In our new approach for synthesis, we utilized Word2Vec and GloVe which hinges on the concept of embedding space. In this space, words are represented as vectors, capturing their semantic relationships based on their context in large text corpora. By embedding the current document and past documents (from the same user) in this space, each word is assigned a position that reflects its semantic meaning.

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/2023-11-09-PersonalizedGeneration_w_LLMAgents/tsne_dim_reduction_example.png" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    An example of how using TSNE dimension reduction can illustrate how words are placed in embedding space. Note that dimension reduction does not always come out cleanly since word embeddings are complex and can't be easily represented in 2D space.
</div>


The ‘centroid’ of the current document in this space is a calculated mean vector, representing the overall semantic direction of the document. Words closest to this centroid are likely to be central to the document’s theme or style. When we look for words from past documents that are closest to this centroid, we are essentially searching for words that align closely with the thematic and stylistic essence of the current document.

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/2023-11-09-PersonalizedGeneration_w_LLMAgents/centroid_example.png" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    An example of how our centroid for the current document corresponds to other words from past documents (note we used PCA dimensionality here). We also chose to display words that had very close euclidean distances to the centroid. Note our centroid from the current document highlighted the following as significant words: ['like', 'since', 'first', 'mystery', 'book']
</div>

This method makes sense technically because it leverages the nuanced understanding of language captured in word embeddings. By focusing on words that are semantically close to the central theme of the current document, the model can more effectively identify and use terms that are likely to be relevant and stylistically consistent for personalization, thereby potentially enhancing the generated text of the personalized generation model.

## Experiment / Word2Vec vs GloVe
Word2Vec and GloVe are both models for word embeddings, but they differ in their approach to creating these embeddings. Word2Vec, developed by Google, primarily uses local context information of words (words surrounding a given word) to generate embeddings. This results in embeddings that capture more of the syntactic and semantic relationships based on specific local contexts.

GloVe (Global Vectors for Word Representation), on the other hand, is designed by Stanford and incorporates global matrix factorization and local context window methods. It emphasizes capturing global statistics of the corpus by considering overall word co-occurrence frequencies, essentially acting as an unsupervised learning algorithm that generates word embeddings.

When used for synthesis in text personalization, these differences influence the nature of the embeddings. Word2Vec might be more sensitive to the specific contextual use of words in the current and past documents, potentially offering more precise thematic matches based on immediate context. GloVe, with its global perspective, might bring in a broader understanding of word use, capturing more general usage patterns and thematic relationships that extend beyond the immediate context. This could lead to a slightly different set of words being selected for personalization in the synthesis process.

In our experiment, we adapted the structure from the "Teach LLMs" paper, incorporating our novel synthesis methods using Word2Vec and GloVe. The process involved independently fine-tuning the personalized generation model for each synthesis approach. This fine-tuning was crucial to observe how the different embedding techniques influenced the model's performance. After implementing the new synthesis methods, we conducted a thorough evaluation to compare their effectiveness, along with the combination of the original and new synthesis approaches, with the base model. The key focus was on analyzing how the different word embeddings (and combinations of embeddings) impacted the quality and personalization of the generated text, with performance metrics providing insights into the strengths and limitations of each method.

# Analysis / Evaluation of Results
The evaluation metrics used in the “Teach LLMs” paper (and also what we utilized), BLEU (Bilingual Evaluation Understudy), ROUGE-1, ROUGE-2, and ROUGE-L, are standard metrics used to evaluate the quality of text which has been machine-translated or generated by machine learning models.

BLEU Score: The BLEU score evaluates the quality of machine-translated text by comparing it with one or more reference translations. It does so at various levels, from individual words to consecutive sequences of words (n-grams), to assess precision. A higher BLEU score indicates more similarity to the reference text, often implying better translation quality. However, BLEU has limitations as it does not account for the fluency or grammaticality of the generated text.

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/2023-11-09-PersonalizedGeneration_w_LLMAgents/bleu_score.png" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Calculations behind the BLEU score calculations.
</div>

ROUGE Scores: ROUGE (Recall-Oriented Understudy for Gisting Evaluation) is another set of metrics for evaluating automatic summarization and machine translation. ROUGE-1 and ROUGE-2 refer to the overlap of unigrams (single words) and bigrams (two consecutive words) between the machine-generated text and a set of reference texts, respectively. ROUGE-L considers the longest common subsequence, focusing on the longest coherently matching sequence of words. ROUGE scores can consider both precision (like BLEU) and recall, providing a more rounded evaluation.

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/2023-11-09-PersonalizedGeneration_w_LLMAgents/rouge_score.png" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Calculations behind the ROUGE-N (N-gram) score calculations; in our case N = 1, 2, or longest common subsequence.
</div>

We can also take a look into how our models performed during the fine tuning period. Based on the progression of the training and validation loss, you can infer how well the model is learning and whether it's overfitting (learning the training data too closely and not generalizing well) or underfitting (not learning the training data well enough).

Comparing the performance of our models using two different synthesis approaches–our base model versus the new synthesis approach using the GloVe or Word2Vec model, and the combination of the base model and new synthesis–could result in different behaviors most likely for one particular reason:

Quality of Embeddings: The GloVe and Word2Vec models provide a different representation for words, capturing semantic relationships in a more nuanced way than just looking at IDF scores, which could lead to varied results during fine tuning. Also, combining our original synthesis with our new synthesis can give the model more information to finetune on allowing for a more intricate understanding of the text when generating.

The differences in BLEU and ROUGE scores between the two models can arise from how each model handles the linguistic nuances of the generated text. If the new approach with the GloVe model is better at capturing the context and meaning of the sentences, it might score higher in BLEU and ROUGE, despite potentially higher loss values.

The variations in BLEU and ROUGE scores could also indicate how each model deals with the trade-off between precision and recall—whether it's better at producing text that contains most of the expected content (high recall) or at avoiding extraneous content not in the reference (high precision).

Evaluating these metrics in combination with each other, rather than in isolation, provides a more comprehensive picture of a model's performance and areas for potential improvement.

The following results portray the overarching BLEU, ROUGE-1, ROUGE-2, and ROUGE-L score we received for the base model, our model using the new synthesis approach, and our model using the base synthesis along with the new synthesis. We have highlighted the snippets of the generated cases that produced our highest scores which are indicative of the possibilities of improvement if we were able to utilize larger T5 models and more training data.

## Results

The following table highlights the results of our evaluation of generated outputs from our baseline model versus our two new approaches (new synthesis and old synth + new synth). Althought there are cases where the max score for our new approaches are high, we believe that this is most likely the case where we generate the rest of a document that is already signficantly short. Essentially, since we don't need to generate a diverse output of words for a longer length, our more compact t5-base model with minimal training performs very well still. [^1]

|                                   	| BLEU (avg) 	| ROUGE1 (avg) 	| ROUGE2 (avg) 	| ROUGEL (avg) 	| BLEU (max) 	| ROUGE1 (max) 	| ROUGE2 (max) 	| ROUGEL (max) 	|
|:----------------------------------	|:-----------:|:-----------:	|:-----------:	|:-----------:	|:-----------:|:-----------:  |:-----------:	|--------------:	|
| Baseline Model                    	| 08.9531    	| 29.5847      	| 18.6126      	| 25.6882      	| 49.5207    	| 65.2174      	| 62.2222      	| 65.2173      	|
| New Synth (Word2Vec)              	| 09.0722    	| 29.3465     	| 18.3129      	| 25.6115      	| 46.6638    	| 65.9340      	| 62.2222      	| 65.2174     	|
| New Synth (GloVe)                 	| 10.3810    	| 31.9870      	| 21.1543      	| 27.4335      	| 50.5317    	| 65.8537      	| 60.1942      	| 63.4146      	|
| New Synth (Word2Vec) + Old Synth  	| 10.4402    	| 31.4181      	| 20.2349      	| 27.7710      	| 58.0197    	| 64.8148      	| 61.9048      	| 62.7907      	|
| New Synth (GloVe) + Old Synth     	| 08.7228    	| 29.2284      	| 17.1685      	| 24.6075      	| 49.7273    	| 65.5462      	| 60.9756      	| 61.9048      	|

[^1]: Output Produced From our Codebase: [https://github.com/dapacica/DL_finalproject_code/blob/main/FinalProjCleanColab.ipynb](https://github.com/dapacica/DL_finalproject_code/blob/main/FinalProjCleanColab.ipynb)

# Conclusion / Discussion of Limitations
Throughout the paper, we have demonstrated the potential of embedding techniques like Word2Vec and GloVe in enhancing the personalization aspect of text generation models. Our experiments, which involved comparing these methods with traditional synthesis techniques, have shown promising results in terms of creating text that more accurately reflects the style and thematic preferences of individual users.

## Limitations
For our exploration, we were limited to running all of our models and doing our data analysis on Google Colab in a short period of time along with having to reimplement the structure used in the "Teach LLMs to Personalize" paper since no codebase exists for it. Because of this, we had to find ways to condense our models and limit the amount of data we ingested so that we could spend less time waiting on models to run and freeing up storage and more time analyzing the output of our code. Two of the big adjustments that we made to navigate these constraints was using the t5-base model (fewer tokens than t5-11b), which we ran for a limited number of epochs, instead of the t5-11b model and using only a subset of data points from the provided Amazon Review Dataset. One of the other things that we tried to make the most advantage of our compute was quantizing our t5-base model to provide faster synthesis and summary to run on our ingested data, but we unfortunately ran into dependency issues and were unable to get this method working. However, from our analysis, we estimate that our evaluation results would have been much more in line with the paper’s results, or even surpass them, if we were able to run the t5-11b model for a larger amount of epochs and utilize more amazon review data.

## Next Steps
If we choose to continue this project, we want to explore ways in which we can synthesize domain-specific knowledge, along with thematic tendencies, related to the current document that can be fed into the final LLM for text generation. There are a lot of benefits of providing synthesized information to the model as it filters for the “most important/significant” words in a document and we hypothesize that this supplementary information could add an extra level of knowledge to a model that has proven to perform well in personalization. 

Also, another pathway that could be explored is integrating Agent LLMs in the initial document ranking phase to see if the procured rankings are better than the current methods set in place (RankDocBM25, RankDocDense, RankSnippet, RankDocBySnpt). We believe that utilizing LLMs that have more awareness of context over large document spaces (and even varying languages) could be benefitial to the process of developing personalized generation model.

# Bibliography
<a id="1">[1]</a> 
Li, Pan, and Alexander Tuzhilin. Towards Controllable and Personalized Review Generation - arXiv.Org, arxiv.org/pdf/1910.03506.pdf. Accessed 12 Dec. 2023. 

<a id="2">[2]</a> 
Li, Junyi, et al. Knowledge-Enhanced Personalized Review Generation with ... - Arxiv.Org, arxiv.org/pdf/2010.01480.pdf. Accessed 12 Dec. 2023. 

<a id="3">[3]</a> 
Gao, Y B, et al. “IOPscience.” Journal of Physics: Conference Series, IOP Publishing, 1 June 2022, iopscience.iop.org/article/10.1088/1742-6596/2294/1/012015. 

<a id="4">[4]</a> 
Li, Cheng, et al. Teach LLMs to Personalize: An Approach Inspired by Writing Education - Arxiv.Org, arxiv.org/pdf/2308.07968.pdf. Accessed 12 Dec. 2023. 