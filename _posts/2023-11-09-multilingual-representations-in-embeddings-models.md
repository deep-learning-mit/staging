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

Recently, [embeddings models](https://platform.openai.com/docs/guides/embeddings) have become incredibly popular as LLMs become more integrated into tools and applications. Embeddings models (specifically, Siamese encoder-only Transformers) are the state-of-the-art method in retrieval, an old problem in computer science. Embeddings are often used in settings like recommendation algorithms, similarity search, and clustering, and have recently found extensive use in Retrieval-Augmented Generation<d-cite key="rag"></d-cite>, assisting LLMs to be more knowledgeable and truthful. However, the best embeddings models are trained on only English data, which means they suffer greatly at applications in other languages, and are inaccessible to most of the world<d-cite key="mteb"></d-cite>. In this blog post, we summarize the history of embeddings research, detail the training regime of a modern embeddings model, present a new multilingual embedding benchmark, and investigate whether it is possible to fine-tune in multilingual capability to a pretrained monolingual model.

Our central question is whether it is possible to learn new languages at the finetuning stage, using contrastive training on publicly available text pair datasets. If successful, it would mean that the encoder can learn a map from one language onto the embedding space of another. This implies that it is possible to approximate translation, at a conceptual level, with a transformation. We will study the results on various language pairs, and compare to a fully pretrained multilingual model.

## The Embedding Task

The aim of embedding text (or any other medium) is to convert human-readable information into vectors. This is useful, because while neural nets cannot process words, images, or sound, they can process vectors. Every NLP model thus has some form of embedding - GPTs, for example, have an embedding layer at the start that transforms input tokens into vector representations<d-cite key="gpt1"></d-cite>. GPTs need an embedding layer because the amount of unique tokens is huge (GPT-2, for example, has 50,257 possible tokens<d-cite key="gpt2"></d-cite>), and it is much more computationally efficient to work with lower-dimensional vectors (GPT-2 embeds these down to 768-dimensional vectors to compute with). 

{% include figure.html path="assets/img/2023-11-09-multilingual-representations-in-embeddings-models/openai_embed.png" class="img-fluid" %}
<div class="caption">
    Embeddings models, as described by OpenAI
</div>

Because of this reduction of information, embeddings are also a form of compression. To turn a whole sentence (or paragraph) into a vector requires prioritising some characteristics and losing others, and we find that the most valuable thing to prioritise is semantic and contextual information. This leads to a very useful property: text pairs with similar meanings or usage patterns tend to have similar vector representations. For example, the vectors "cat" and "dog" are closer to each other than "cat" and "cucumber". Even more interestingly, as found in the Word2Vec paper, this property causes embeddings to have arithmetic consistency, as shown in the famous "king - man + woman = queen" example.<d-cite key="w2v"></d-cite> You can explore the Word2Vec embedding space in the interactive visualization below:

<div class="l-page">
  <iframe src="{{ 'assets/html/2023-11-09-multilingual-representations-in-embeddings-models/word2vec_demo.html' | relative_url }}" frameborder='0' scrolling='no' height="600px" width="100%"></iframe>
</div>
<div class="caption">
    Visualisation of Word2Vec for the 250 most common English nouns
</div>

While this may seem abstract, embeddings have found usage in many downstream and commercial tasks, including:

1. **Classification** - embeddings models classify sentences, such as in sentiment analysis between positive or negative airline reviews<d-cite key="sent"></d-cite>. 
2. **Search** - models return nearest-embedded results to a search query, understanding synonyms and context<d-cite key="sgpt"></d-cite>.
3. **Recommendation** - models return embeddings that suggest related items users may like, for example [clothes and jewellery](https://arxiv.org/pdf/1507.08439.pdf).
4. **Clustering** - embeddings are used to cluster datapoints into smaller groups, with downstream algorithms like k-means<d-cite key="kmeans"></d-cite>.
5. **Reranking** - embeddings are used to sort a list, such as one retrieved from a database, into most relevant items<d-cite key="rerank"></d-cite>.
6. **Retrieval** - a query is embedded, and answers are selected by the closeness of their embedding.<d-cite key="beir"></d-cite>.

### History and Background

The first successful approaches to these problems were bag-of-words models. These are non-neural algorithms that work by ranking documents based on how many word occurrences they share. There were some improvements around this basic idea, for example Okapi BM25<d-cite key="bm25"></d-cite> includes a term for the expected likelihood of that word co-occurring.

<table>
  <tr>
    <th>Sentence</th>
    <th>about</th>
    <th>bird</th>
    <th>bird,</th>
    <th>heard</th>
    <th>is</th>
    <th>the</th>
    <th>word</th>
    <th>you</th>
  </tr>
  <tr>
    <td>About the bird, the bird, bird bird bird</td>
    <td>1</td>
    <td>3</td>
    <td>2</td>
    <td>0</td>
    <td>0</td>
    <td>2</td>
    <td>0</td>
    <td>0</td>
  </tr>
  <tr>
    <td>You heard about the bird</td>
    <td>1</td>
    <td>1</td>
    <td>0</td>
    <td>1</td>
    <td>0</td>
    <td>1</td>
    <td>0</td>
    <td>1</td>
  </tr>
  <tr>
    <td>The bird is the word</td>
    <td>0</td>
    <td>1</td>
    <td>0</td>
    <td>0</td>
    <td>1</td>
    <td>2</td>
    <td>1</td>
    <td>0</td>
  </tr>
</table>
<div class="caption">
    A table demonstrating bag-of-words calculation.
</div>

The first neural approaches to this problem actually used bag-of-words as a loss function, for example Word2Vec (2013)<d-cite key="w2v"></d-cite> used either continuous bag-of-words (CBOW) or skipgram loss to train a word embedding model. Word2Vec itself is a shallow two-layer neural network that is used to generate an embedding, which in the CBOW training regime is used to predict a word given a bag of surrounding words. The skipgram loss is similar, but weighs words depending on their proximity to the word we're trying to predict.  This word-prediction-from-embeddings task is a key facet of training language models to have useful representations, and we'll see it again later. 

Word2Vec had some incredible results, and was later improved by subsequent approaches<d-cite key="glove"></d-cite>, but word embeddings often failed due to the fact that words with multiple meanings had to share the same point in the embedding space. The sentences "I went to the bank to cash a check" and "I went to the bank to catch a fish" are obviously semantically unrelated, but the word "bank" will necessarily have to share an embedding, making the embedding itself likely meaningless.  

<div class="l-page">
  <iframe src="{{ 'assets/html/2023-11-09-multilingual-representations-in-embeddings-models/special_demo.html' | relative_url }}" frameborder='0' scrolling='no' height="600px" width="100%"></iframe>
</div>
<div class="caption">
    Visualisation of Word2Vec struggling with polysemanticity in the "riverbank" example
</div>

To solve this, embeddings had to be generated in-context, and be able to support multiple meanings. There were some attempts at changing Word2Vec to support polysemanticity, such as  Multi-Sense Skip-Gram (MSSG)<d-cite key="mssg"></d-cite>, but they required hacky workarounds such as pre-programming an expected number of meanings for each word. 

#### BERT

BERT<d-cite key="bert"></d-cite> was arguably the beginning of the LLM revolution, as it showed for the first time that a single pretrained language model could be finetuned to support many different tasks downstream. It was essentially an embeddings model - trained again with the word prediction task, now with the context of words not weighted by proximity, but by a trainable position embedding that provided information that the model could use to predict long-term associations and causality. It produced both word-level and sentence-level embeddings, that proved extraordinarily useful for the embeddings tasks described above.

##### BERT Training

{% include figure.html path="assets/img/2023-11-09-multilingual-representations-in-embeddings-models/bert.png" class="img-fluid" %}
<div class="caption">
    BERT architecture diagram
</div>

BERT (Bidirectional Encoder Representations from Transformers) was based on the Transformer architecture introduced by Vashwani et al. in 2017<d-cite key="attn"></d-cite>. The key differences were that BERT was allowed bidirectional context rather than left-side-only, that it did not include a decoder, and its masked language modeling and next sentence prediction training objectives.

{% include figure.html path="assets/img/2023-11-09-multilingual-representations-in-embeddings-models/mlm.png" class="img-fluid" %}
<div class="caption">
    BERT's Masked Language Modeling loss
</div>

MLM works by taking 15% of the text tokens that BERT sees and replacing them with a [MASK] token. The model's objective is to predict that masked word with its embedding, using the context from the surrounding tokens, and then it is trained on the cross-entropy loss between the predictions and the actual truth.

BERT was also trained on the NSP (Next Sentence Prediction) objective. In training, the model is given a pair of input segments, and its task is to predict whether the second segment (segment B) follows the first one (segment A) in the original text or if they are randomly sampled and unrelated. The input is constructed by concatenating segment A, which is preceded by a special [CLS] token, and segment B, with a special [SEP] (separator) token in between. For example: "[CLS] Segment A [SEP] Segment B". BERT then produces a pair of embeddings: one for the [CLS] token at the beginning of the input and one for the [SEP] token that separates the two segments. These embeddings are then used to compute a binary classification. The intended effect is that [CLS] contains information about the overall meaning of the first sentence, and [SEP] contains information about the second. This is the first example of sentence embeddings, which are the key to how a modern embeddings model works.

{% include figure.html path="assets/img/2023-11-09-multilingual-representations-in-embeddings-models/nsp.png" class="img-fluid" %}
<div class="caption">
    BERT's Next Sentence Prediction loss
</div>

BERT turns token inputs into embeddings for each token in its context window, which is 512 tokens long. We can choose to construct a single text embedding from this any way we like. There are several popular strategies for this "token pooling" problem. Reading the above, one may be tempted to take the [CLS] token's embedding. In practice, however, the [CLS] token embeddings proved to be slightly worse than just taking the average of all the individual token embeddings of the sentence<d-cite key="berthater"></d-cite>, and subsequent models such as RoBERTa<d-cite key="roberta"></d-cite> skipped the NSP training objective and actually performed slightly better. Why this is the case is an area of ongoing research, but as a matter of opinion, we personally suspect Shitao Xiao's work on RetroMAE<d-cite key="rmae"></d-cite> correctly diagnoses the issue, as demonstrated by their models' improved performance on benchmarks. The training losses described in that paper are more complex and outside the scope of this blog post, but it's worth a read if interested. 

#### SBERT

The final part of the story is Sentence-BERT<d-cite key="sbert"></d-cite>, and its addition of contrastive text-pair pretraining. This what turns BERT, a general language model, into a model that specifically generates text embeddings. Contrastive training was discussed at length in 6.s898; the core insight is that we can train an encoder model to have a useful representation if we train it to embed similar examples together, and dissimilar examples far apart. In Sentence Transformers, this is done by contructing a "Siamese BERT" network. There are two BERT models (or commonly two copies of the same model) that are each used to embed a text passage. Then, the loss is calculated by the following formula:

$$
\mathcal{L}_N = -\mathbb{E}_{X} \left[ \log \frac{f_k(x_t+k, c_t)}{\sum_{x_j \in X} f_k(x_j, c_t)} \right]
$$

This encourages the model to predict positive pairs (similar passages) as vectors with close to 1 similarity, and negative pairs close to 0. Similarity metrics include (Euclidean) distance, but most often used is cosine similarity. Negative pairs can either be "mined" with some heuristic such as bag-of-words, or simply sampled at random from other examples in the batch. Due to this, pretraining batch sizes for embedding BERTs are often huge, in the tens of thousands<d-cite key="gte"></d-cite>. 

{% include figure.html path="assets/img/2023-11-09-multilingual-representations-in-embeddings-models/sbert.png" class="img-fluid" %}
<div class="caption">
    The Siamese BERT architecture
</div>

The reason two models are used is that many tasks see improved performance if there is a distinction made between "questions" and "answers". For example, searches and retrieval queries may not resemble the results they most need in meaning: "What is the the tallest building Hong Kong" and "The International Commerce Centre" are not closely semantically related, but should be paired in search contexts. Because of this, we can train a "query" and "passage" model together as one giant network on a contrastive loss, and thus get a model that can take in both. 

In practice, this improvement is rarely worth doubling the number of parameters, and so most papers simply re-use the same model for both queries and passages.

## How Embeddings Models are Trained

Putting all this together, we have the current standard recipe for training a modern embeddings model, in up to three stages:

### 1. Pretraining

It is valuable to start with a language model that has already learned some inner representation of language. This makes the embeddings task significantly easier, since the model must only learn to condense this inner representation into a single high-dimensional dense vector space. While it is possible to use more modern LLMs such as GPT or LLaMA for embeddings<d-cite key="sgpt"></d-cite>, they are fundamentally hampered because they cannot attend to context in both directions. Therefore, almost all state-of-the-art embeddings models begin from the BERT models themselves, or their derivatives<d-cite key="gte"></d-cite><d-cite key="e5"></d-cite>. These are trained as described above, with an MLM and potentially NSP loss.

### 2. Training

Following Sentence-BERT, the model is trained contrastively. At this point, we can choose a pooling strategy to convert BERT outputs into sentence embeddings. Many current papers choose to use average pooling<d-cite key="e5"></d-cite>. Positive pairs are either handpicked from datasets such as search engine question-responses, or commonly generated from general text data, such as academic paper title-abstract pairs, Wikipedia page title-summaries and so forth<d-cite key="gte"></d-cite><d-cite key="bge"></d-cite>. 

### 3. Finetuning

It has also become common to fine-tune especially large embeddings models on higher-quality datasets, such as MS MARCO (Bing question-passage responses), fact verification (e.g. FEVER), and paraphrasing (e.g. Quora). This increases performance at desired tasks<d-cite key="bge"></d-cite>.

## How Embeddings Models are Tested

Similarly to how decoder LLMs have recently converged on being measured on the HuggingFace Open LLM Leaderboard, the currently ubiquitous benchmark for embeddings models is MTEB<d-cite key="mteb"></d-cite>. Presented in a 2022 paper, it contains 8 embedding tasks covering a total of 58 datasets. The tasks are:


{% include figure.html path="assets/img/2023-11-09-multilingual-representations-in-embeddings-models/mteb.png" class="img-fluid" %}
<div class="caption">
    MTEB datasets
</div>


1. **Bitext Mining**:
Inputs are two sets of sentences from two different languages. For each sentence in the first set, the best match in the second set needs to be found. This metric is commonly ignored in places such as the MTEB Leaderboard and in papers, because few multilingual models have been created.

2. **Classification**:
A train and test set are embedded with the provided model. The train set embeddings are used to train a logistic regression classifier, which is scored on the test set.

3. **Clustering**: Involves grouping a set of sentences or paragraphs into meaningful clusters. A k-means model is trained on embedded texts. The model's performance is assessed using the v-measure, which is independent of the cluster labels.

4. **Pair Classification**: Requires assigning labels to pairs of text inputs, typically indicating if they are duplicates or paraphrases. Texts are embedded and distances calculated using various metrics (cosine similarity, dot product, Euclidean, Manhattan). Metrics like accuracy, average precision, F1, precision, and recall are used.

5. **Reranking**: Involves ranking query results against relevant and irrelevant reference texts. Texts are embedded using a model, with cosine similarity determining relevance. Rankings are scored using mean MRR@k and MAP, with MAP as the primary metric.

6. **Retrieval**: Each dataset includes a corpus and queries, with a goal to find relevant documents. Models embed queries and documents, computing similarity scores. Metrics like nDCG@k, MRR@k, MAP@k, precision@k, and recall@k are used, focusing on nDCG@10.

7. **Semantic Textual Similarity (STS)**: Involves assessing the similarity of sentence pairs. Labels are continuous, with higher scores for more similar sentences. Models embed sentences and compute similarity using various metrics, benchmarked against ground truth using Pearson and Spearman correlations. Spearman correlation based on cosine similarity is the main metric.

8. **Summarization**: Evaluates machine-generated summaries against human-written ones. Models embed summaries, computing distances between machine and human summaries. The closest score, such as the highest cosine similarity, is used for evaluation. Metrics include Pearson and Spearman correlations with human assessments, focusing on Spearman correlation based on cosine similarity.

We can see that MTEB represents many downstream users' desires as described earlier, but could be criticised for favoring cosine similarity as a distance metric for training. In either case, MTEB has demonstrated, and itself encouraged, some trends in research:

### Scaling

The MTEB paper itself, as well as the GTR<d-cite key="gtr"></d-cite> and Sentence-T5<d-cite key="st5"></d-cite> papers, suggested that model parameters are correlated with higher performance. We should expect that from intuition about GPTs, larger models perform better<d-cite key="chinchilla"></d-cite>. 

{% include figure.html path="assets/img/2023-11-09-multilingual-representations-in-embeddings-models/scaling.png" class="img-fluid" %}
<div class="caption">
    Figure 3 from MTEB demonstrating scaling vs. performance
</div>

However, if we extrapolate to more recent research , we find that the state-of-the-art models have failed to get bigger over time, and the highest-performance models are still under 1B parameters. This shows that embeddings is not as easily reduced to scaling laws as LLMs are. 

{% include figure.html path="assets/img/2023-11-09-multilingual-representations-in-embeddings-models/scale.png" class="img-fluid" %}
<div class="caption">
    MTEB score vs time for SOTA models. The size of the cross represents parameter count.
</div>

Even the small models still train on hundreds of millions or billions of text pairs<d-cite key="gtr"></d-cite>, requiring thousands of GPU-hours to train. We can conclude that while parameter count may not be increasing, the overall compute requirements of training an embeddings model are getting higher and higher, and it is no longer within the reach of all researchers to work on these models.


### Multilingualism

While MTEB is a multilingual benchmark, only a few tasks, namely STS, Classification and Bitext Mining, have multilingual versions. Combined with the abundance of English training data, this has led to every language except English, Chinese and Polish lacking a complete MTEB and thus lacking the benefits of state-of-the-art models.

{% include figure.html path="assets/img/2023-11-09-multilingual-representations-in-embeddings-models/langs.png" class="img-fluid" %}
<div class="caption">
    MTEB SOTA vs. language. Languages other than Chinese and English lack the benefits of progress.
</div>

## Finetuning In Multilingualism

With these problems as our motivation, we aim to find out if it is possible to add multilingualism to an existing model without having to pretrain from scratch. This may be a step towards bringing the benefits of increased embeddings performance to languages that don't currently have a state-of-the-art model. Furthermore, if it is possible to add a new language to an existing model, this hints at the ideas that models do not necessary learn a representation based on a particular language, and that translation is easier than expected in the context of embeddings.

To do this, we will take an existing model that has both monolingual English and multilingual variants, and try low-cost methods to add in new languages without sacrificing English performance. We will attempt to create a model that performs on-par with the multilingual model in multiple languages, and on-par with the original model in English, which we will measure by completing with our own data a multilingual version of MTEB in all tasks.

### Model Choice

We choose e5-base-v2 and e5-base-multilingual<d-cite key="e5"></d-cite> as our test models. E5 is the highest-performing current open-weights model with both a mono- and multilingual version, and still holds the top spot in many languages. Both models are the size of BERT, with 12 layers, 768-dimensional embeddings and a context window of 512 tokens. The only difference is that the multilingual model has a much larger vocabulary to support more languages, and uses the XLM-RoBERTa tokenizer, leading to about 60% more parameters. 

This choice does produce a caveat in the rest of our post - since the tokenizer at the start of e5-base has been trained only on English data, it will be unable to tokenize text that is not also a possible English string. In practice, this means that any Latin or near-Latin speaking languages, such as French, German and Turkish, can be used, but the model cannot be finetuned to read Japanese or Arabic script. Any non-Latin characters will likely become an [UNK] token, which carries no information for the model to embed. We are confident that this is not a fatal flaw, though, since just as it is possible to train LLMs with unused vocabulary, such as Persimmon-8B<d-cite key="persimmon"></d-cite>, it is possible to train an embeddings model with a big unused vocabulary. In the case that this research proves useful, it would be possible to train an English embeddings model with a multilingual tokenizer and fill in this extra vocabulary space in finetuning.

## Method
### Benchmarking

As described above, it is hard to use MTEB to test performance in non-English languages, due to the lack of available tasks. After investigating the source datasets, we know that this is because of a lack of data. In the interest of producing a universally fair test, especially for low-resource languages, we opted to use synthetic data to create the multilingual MTEB test set, by translating the English sets into each language.

<div style="margin-top: 0.5em; margin-bottom: 1em; padding: 1em; background-color: #f2f5f7; border-radius: 10px; font-size: 1rem">
<i>Side note: We were fascinated to find that the state-of-the-art neural machine translation model is no longer GNMT<d-cite key="gnmt"></d-cite> or the Google Translate API, but in fact just GPT-4!</i>
</div>

We used GPT 3.5 and Mixtral-8x7B to process 400K examples in each of our five languages, to produce the final benchmark. Running our models we can see:
