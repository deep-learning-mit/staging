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

## Introduction

Recently, [embeddings models](https://platform.openai.com/docs/guides/embeddings) have become incredibly popular as LLMs become integrated into tools and applications. Embeddings models (specifically, Siamese encoder-only Transformers) are the state-of-the-art method in the retrieval task, an old problem in computer science. Embeddings are often used in settings like recommendation algorithms, similarity search, and clustering, and have recently found extensive use in Retrieval-Augmented Generation<d-cite key="rag"></d-cite>, assisting LLMs to be more knowledgeable and truthful. However, many of the most high-performance embeddings models are trained on English only, which means they suffer greatly at applications in other languages, and are inaccessible to most of the world<d-cite key="mteb"></d-cite>. In this blog post, we summarize the history of embeddings research, detail the training regime of a modern embeddings model, present a new multilingual embedding benchmark, and investigate whether it is possible to fine-tune in multilingual capability to a pretrained monolingual model.

Our central question is whether it is possible to insert new languages at the finetuning stage, using contrastive training on publicly available datasets of text pairs. If successful, it would mean that the encoder can learn a map from one language onto the embedding space of another. This implies that it is possible to approximate translation, at a conceptual level, with a transformation. We will study the results on various language pairs, and compare to a fully pretrained multilingual model.

## The Embedding Task

The aim of embedding text (or any other medium) is to convert human-readable information into vectors. This is useful, because while neural nets cannot process words, images, or sound, they can process vectors. Every NLP model thus has some form of embedding - GPTs, for example, have an embedding layer at the start that transforms input tokens into vector representations. GPTs need an embedding layer because the amount of unique tokens is huge (GPT-2, for example, has 50,257 possible tokens), and it is much more computationally efficient to work with lower-dimensional vectors (GPT-2 embeds these down to 768-dimensional vectors to compute with). 

Because of this reduction of information, embeddings are also a form of compression. To turn a whole sentence (or paragraph) into a vector requires prioritising some characteristics and losing others, and we find that the most valuable thing to prioritise is semantic and contextual information. This leads to a very useful property: text pairs with similar meanings or usage patterns tend to have similar vector representations. For example, the vectors "cat" and "dog" are closer to each other than "cat" and "cucumber". Even more interestingly, as found in the Word2Vec paper, this property causes embeddings to have arithmetic consistency, as shown in the famous "king - man + woman = queen" example. 

Embeddings have found usage in these downstream tasks:
1. Classification
2. Search
3. Recommendation
4. Clustering
5. Reranking
6. Retrieval

### History and Background

The first successful approaches to these problems were bag-of-words models. These are non-neural algorithms that work by ranking documents based on how many word occurrences they share. There were some improvements around this basic idea, for example Okapi BM25 includes a term for the expected likelihood of that word co-occurring.

The first neural approaches to this problem actually used bag-of-words as a loss function, for example Word2Vec (2013) used either continuous bag-of-words (CBOW) or skipgram loss to train a word embedding function. Word2Vec itself is a shallow two-layer neural network that is used to generate an embedding, which in the CBOW training regime is used to predict a word given a bag of surrounding words. This word-prediction-from-embeddings task is a key facet of training language models to have useful representations, and we'll see it again later. The skip-gram loss is similar, but weighs words depending on their proximity to the word we're trying to predict. 

Word2Vec had some incredible results, and was later improved by subsequent approaches, but word embeddings often failed due to the fact that words with multiple meanings had to share the same point in the embedding space. The sentences "I went to the bank to cash a check" and "I went to the bank to catch a fish" are obviously semantically unrelated, but the word "bank" will necessarily have to share an embedding, making the embedding itself likely meaningless.  

To solve this, embeddings had to be generated in-context, and be able to support multiple meanings. There were some attempts at changing Word2Vec to support polysemanticity, such as  Multi-Sense Skip-Gram (MSSG), but they required hacky workarounds such as pre-programming an expected number of meanings for each word. 

#### BERT

BERT was arguably the beginning of the LLM revolution, as it showed for the first time that a single pretrained language model could be finetuned to support many different tasks downstream. It was essentially an embeddings model - trained again with the word prediction task, now with the context of words not weighted by proximity, but by a trainable position embedding that provided information that the model could use to predict long-term associations and causality. It produced both word-level and sentence-level embeddings, that proved extraordinarily useful for the embeddings tasks described above.

##### BERT Training

include figure.html path="assets/img/2023-11-09-multilingual-representations-in-embeddings-models/bert.png" class="img-fluid" 

BERT (Bidirectional Encoder Representations from Transformers) was based on the Transformer architecture introduced by Vashwani et al. in 2017. The key differences were that BERT was allowed bidirectional context rather than left-side-only, that it did not include a decoder, and its masked language modeling and next sentence prediction training objectives.

include figure.html path="assets/img/2023-11-09-multilingual-representations-in-embeddings-models/mlm.png" class="img-fluid"

MLM works by taking 15% of the text tokens that BERT sees and replacing them with a [MASK] token. The model's objective is to predict that next word with that token's embedding, using the context from the surrounding tokens, and then it is trained on the cross-entropy loss between the predictions and the actual truth.

BERT was also trained on the NSP (Next Sentence Prediction) objective. In training, the model is given a pair of input segments, and its model's task is to predict whether the second segment (segment B) follows the first one (segment A) in the original text or if they are randomly sampled and unrelated. The input is constructed by concatenating segment A, which is preceded by a special [CLS] token, and segment B, with a special [SEP] (separator) token in between. For example: "[CLS] Segment A [SEP] Segment B [SEP]". BERT then produces a pair of embeddings: one for the [CLS] token at the beginning of the input and one for the [SEP] token that separates the two segments. These embeddings are then used to compute a binary classification. The intended effect is that [CLS] contains information about the overall meaning of the first sentence, and [SEP] contains information about the second. This is the first example of sentence embeddings, which are the key to how a modern embeddings model works.

BERT turns token inputs into embeddings for each token in its context window, which is 512. We can choose to construct a single text embedding from this in a few ways, which is called "token pooling". Reading the above, one may be tempted to take the [CLS] token's embedding alone. In practice, however, the [CLS] token embeddings proved to be slightly worse than just taking the average of all the individual token embeddings of the sentence, and subsequent models such as RoBERTa skipped the NSP training objective and actually performed slightly better. Why this is the case is an area of ongoing research, but as a matter of opinion, we personally suspect Shitao Xiao's work on RetroMAE and DupMAE correctly diagnoses the issue, as demonstrated by their models' improved performance on benchmarks. The training losses described in those papers are more complex and outside the scope of this blog post, but it's worth a read if interested. 

#### SBERT

The final part of the story is Sentence-BERT, and its addition of contrastive text-pair pretraining. This what turns BERT, a general language model, into a model that specifically generates text embeddings. Contrastive training was discussed at length in 6.s898; the core insight is that we can train an encoder model to have a useful representation if we train it to embed similar examples together, and dissimilar examples far apart. In Sentence Transformers, this is done by contructing a "Siamese BERT" network. There are two BERT models (or commonly two copies of the same model) that are each used to embed a text passage. Then, the loss is calculated by the following formula:

This encourages the model to predict positive pairs (similar passages) as vectors with close to 1 similarity, and negative pairs close to 0. Similarity metrics include (Euclidean) distance, but most often used is cosine similarity. Negative pairs can either be "mined" with some heuristic such as bag-of-words, or simply sampled at random from other examples in the batch. Due to this, pretraining batch sizes for embedding BERTs are often huge, in the tens of thousands. 

The reason two models are used is that many tasks mentioned at the beginning improve performance if there is a distinction made between "questions" and "answers". For example, searches and retrieval queries may not resemble the results they most need in meaning: "What is the the tallest building Hong Kong" and "The International Commerce Centre" are not closely semantically related, but should be paired in search contexts. Because of this, we can train a "query" and "passage" model together as one giant network on a contrastive loss, and thus get a model that can take in both. 

In practice, this improvement is not worth doubling the number of parameters, and so most modern papers simply re-use the same model for both queries and passages.

## How Embeddings Models are Trained

Putting all this together, we have the standard recipe for training a modern embeddings model, in three stages:

### 1. Pretraining

It is valuable to start with a language model that has already learned some inner representation of language. This makes the embeddings task significantly easier, since the model must only learn to condense this inner representation into a single high-dimensional dense vector space. While it is possible to use more modern LLMs such as GPT or LLaMA for embeddings, they are fundamentally hampered because they cannot attend to context in both directions. Therefore, almost all state-of-the-art embeddings models begin from the BERT models themselves, or their derivatives. These are trained as described above, with an MLM and potentially NSP loss.

### 2. Training

Following Sentence-BERT, the model is trained contrastively. At this point, we can choose a pooling strategy to convert BERT outputs into sentence embeddings. Many current papers choose to use average pooling. Positive pairs are either handpicked from datasets such as search engine question-responses, or commonly generated from general text data, such as academic paper title-abstract pairs, Wikipedia page title-summaries and so forth. 

### 3. Finetuning

It has also become common to fine-tune especially large embeddings models on higher-quality datasets, such as MS MARCO (Bing question-passage responses), fact verification (e.g. FEVER), and paraphrasing (e.g. Quora). This increases performance at desired tasks.

## How Embeddings Models are Tested

Similarly to how decoder LLMs have recently converged on being measured on the HuggingFace Open LLM Leaderboard, the currently ubiquitous benchmark for embeddings models is MTEB. Presented in a 2022 paper, it contains 8 embedding tasks covering a total of 58 datasets. The tasks are:

1. Bitext Mining 
Inputs are two sets of sentences from two different languages. For each sentence in the first set, the best match in the second set needs to be found. This metric is commonly ignored in places such as the MTEB Leaderboard and in papers, because few multilingual models have been created.

2. Classification 
A train and test set are embedded with the provided model. The train set embeddings are used to train a logistic regression classifier, which is scored on the test set.

3. Clustering: Involves grouping a set of sentences or paragraphs into meaningful clusters. A k-means model is trained on embedded texts. The model's performance is assessed using the v-measure, which is independent of the cluster labels.

4. Pair Classification: Requires assigning labels to pairs of text inputs, typically indicating if they are duplicates or paraphrases. Texts are embedded and distances calculated using various metrics (cosine similarity, dot product, Euclidean, Manhattan). Metrics like accuracy, average precision, F1, precision, and recall are used.

5. Reranking: Involves ranking query results against relevant and irrelevant reference texts. Texts are embedded using a model, with cosine similarity determining relevance. Rankings are scored using mean MRR@k and MAP, with MAP as the primary metric.

6. Retrieval: Each dataset includes a corpus and queries, with a goal to find relevant documents. Models embed queries and documents, computing similarity scores. Metrics like nDCG@k, MRR@k, MAP@k, precision@k, and recall@k are used, focusing on nDCG@10.

7. Semantic Textual Similarity (STS): Involves assessing the similarity of sentence pairs. Labels are continuous, with higher scores for more similar sentences. Models embed sentences and compute similarity using various metrics, benchmarked against ground truth using Pearson and Spearman correlations. Spearman correlation based on cosine similarity is the main metric.

8. Summarization: Evaluates machine-generated summaries against human-written ones. Models embed summaries, computing distances between machine and human summaries. The closest score, such as the highest cosine similarity, is used for evaluation. Metrics include Pearson and Spearman correlations with human assessments, focusing on Spearman correlation based on cosine similarity.

We can see that MTEB represents many downstream users' desires as described earlier, but could be criticised for favoring cosine similarity as a distance metric. In either case, MTEB has demonstrated, and itself encouraged, some trends in research:

### Scaling

The MTEB paper itself, as well as GTE and ST5 papers, suggested that model parameters are correlated with higher performance. We should expect that from intuition about GPTs, larger models perform better. 

However, if we extrapolate the graph, we find that the state-of-the-art models have failed to get bigger over time, and the highest-performance models are still under 1B parameters. The key difference is that these models have been trained over extremely large amounts of data. For the models we can get text pair information about, we can see the trend increasing.

We can conclude that while parameter count may not be increasing, the overall compute requirements of training an embeddings model are getting higher and higher, and it is no longer within the reach of all researchers to work on these models.

### Multilingualism

While MTEB is a multilingual benchmark, only a few tasks, namely STS, Classification and Bitext Mining, have multilingual versions. Combined with the abundance of English training data, this has led to every language except English, Chinese and Polish lacking a complete MTEB and thus lacking the benefits of state-of-the-art models.

## Finetuning In Multilingualism

With these problems as our motivation, we aim to find out if it is possible to add multilingualism to an existing model without having to pretrain from scratch. This may be a step towards bringing the benefits of increased embeddings performance to languages that don't currently have a state-of-the-art model. Furthermore, if it is possible to add a new language to an existing model, this hints at the ideas that models do not necessary learn a representation based on a particular language, and that translation is easier than expected in the context of embeddings.

To do this, we will take an existing model that has both monolingual English and multilingual variants, and try low-cost methods to add in new languages without sacrificing English performance. We will attempt to create a model that performs on-par with the multilingual model in multiple languages, and on-par with the original model in English, which we will measure by completing with our own data a multilingual version of MTEB in all tasks.

### Model Choice

We choose e5-large-v2 and e5-large-multilingual as our test models. E5 is the highest-performing current model with both a mono- and multilingual version, and still holds the top spot in many languages. Both models are the size of bert-large, with 

<!-- 
## Ideas \[proposal\]

As stated above, the goal is to use embeddings models as a reason to explore whether it is possible to efficiently teach a Siamese BERT<d-cite key="sbert"></d-cite> a new language. There is already plenty of work in this area, with Transformers<d-cite key="attn"></d-cite> themselves being initially applied to the task of Neural Machine Translation<d-cite key="nmt"></d-cite>. However, the contrastive training means that we would not be directly training a translation model, but encouraging an observable map of one language onto another. This means we can make plenty of visualisations, since the embeddings space lends itself very naturally to things like tSNE reduction and visualisation.

We can also discuss the effects that contrastive training has on the embeddings space - as mentioned in PSET 4, an accurate encoder tends to produce many small, tight class clusters on the output sphere. It would be interesting to see if the introduction of a new language results in the creation of new clusters, or if they are integrated into the already-existing ones. 

There are some unresolved questions about the tokenizer. In "easier" language pairs, such as Spanish-English or French-English, an English tokenizer can handle (albeit inefficiently) the new text. In pairs like Japanese-English, we would have to extend the BERT/e5 tokenizer, which would potentially damage performance. We have the compute to train an English embeddings model from scratch in a few days' cluster time, which could be a solution for these harder pairs.

## Methods \[proposal\]

We already have lots of experience working with embeddings models together. We have done a lot of research on them already, and have some preliminary experiments suggesting this project would be a success. 

There is a good candidate pair of models for this blog post: [e5-base](https://huggingface.co/intfloat/e5-base-v2) and [multilingual-e5-base](https://huggingface.co/intfloat/multilingual-e5-base) share the same architecture and were trained by the same team, showing roughly equivalent performance modulo the expected multilingual hit to English performance. Graciously, the code is also public at [unilm/e5](https://github.com/microsoft/unilm/tree/master/e5). Datasets of multilingual text pairs include [CCMatrix](https://ai.meta.com/blog/ccmatrix-a-billion-scale-bitext-data-set-for-training-translation-models/) and [NLLB](https://huggingface.co/datasets/allenai/nllb). We aim to run experiments in the format of training an English model such as e5 on large datasets of text pairs with another language, and comparing that model's performance to the pretrained e5-multilingual model. There are also several multilingual benchmarks on this task that we could use to quantify this, such as [Mr. TyDi](https://arxiv.org/abs/2108.08787), and [MTEB's English, Chinese and Polish variants](https://huggingface.co/blog/mteb).

Sadly, the code for contrastively training e5 is not public. However, [Sentence Transformers](https://www.sbert.net/docs/training/overview.html)<d-cite key="sbert"></d-cite> is a good library for contratively training BERTs, and is built on top of Huggingface Transformers. We are also writing our own version of this libary in MosaicML Composer for our HPC work, due to its easier data ingress.

As well as using e5, we are also in the midst of pretraining our own multilingual embeddings model with significantly better performance, and a longer context length. This is achieved through lots of hacks (ALiBi, muP, Flash), as well as a new pretraining objective, RetroMAE<d-cite key="rmae"></d-cite>. We hope this model will be ready to use in the blog post, but the project does not require it to continue, and it won't slow us down either way.

Compute is not a concern. We have access to 7,500 A100-hours (>1000 exaFLOP) for this project, thanks to a compute grant from Stability AI. We also have access to an unlimited s3 bucket, paid for by the Stability grant, in which we can store and stream data. The language pairs datasets mentioned are in the billions, so we can expect terabytes of training data. -->