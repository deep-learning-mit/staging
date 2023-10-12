---
layout: distill
title: Transformers Learn Faster by Taking Notes on the Fly
description: If your unsupervised pre-training is taking forever and you need a lightweight solution that will accelerate it, taking notes might be the method you are looking for! This method takes notes of the contextual information of the rare words and incorporates this information as a part of their embeddings on the fly! The solution is lightweight in the sense that it does not increase the inference time and it does not require an additional pass during training. The experiments demonstrate that this method reduces the pre-training time of large language models by up to 60%.
date: 2023-02-07
htmlwidgets: true

# Anonymize when submitting
# authors:
#   - name: Anonymous

authors:
  - name: Anonymous
    url: "https://en.wikipedia.org/wiki/Albert_Einstein"
    affiliations:
      name: Anonymous
  - name: Anonymous
    url: "https://en.wikipedia.org/wiki/Boris_Podolsky"
    affiliations:
      name: Anonymous

# must be the exact same name as your blogpost
bibliography: 2023-02-07-taking_notes_on_the_fly.bib  

# Add a table of contents to your post.
#   - make sure that TOC names match the actual section names
#     for hyperlinks within the post to work correctly.
toc:
  - name: Introduction
  - name: Background
    subsections:
    - name: Transformers
    - name: Word Distribution in Texts
  - name: Related Work
  - name: Methodology
  - name: Experiments
  - name: Conclusion


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

## Introduction
{% include figure.html path="assets/img/2023-02-07-taking_notes_on_the_fly/example.png" class="img-fluid" %}
<div class="caption">
   Figure 1 from the paper, which describes an example situation in which taking notes could be useful. Here COVID is a rare word. Therefore, the model is struggling with the completion task on the left because it has not possibly seen many sentences with the word COVID in it. Thus, we take notes of the contextual information of it as we see examples on it in the training set and we quickly learn to associate it with which words! 
</div>

Transformers, which were invented by Google in 2017 <d-cite key="vaswani2017attention"></d-cite>, have become the go-to architecture for various tasks in many domains, such as natural language processing and computer vision <d-cite key="dosovitskiy2020image" ></d-cite>, <d-cite key="devlin2018bert" ></d-cite>, <d-cite key="radford2018improving" ></d-cite>, <d-cite key="brown2020language" ></d-cite>. The success of transformers are mainly because they have two amazing properties: 

1. They are phenomenal in grasping the context of words within the bodies of text that they belong to.

2. They do not process the input sequences in order. Thus, their operations can easily be parallelized. 

Equipped with these powerful features, transformers have excelled in unsupervised pre-training tasks, which is the driving force of several state-of-the-art models, such as BERT and GPT-3. In unsupervised pre-training, a large and diverse dataset is used to train the (baseline) model. If someone wishes to fine-tune the base model for a specific task, they can do so by training it with a relatively smaller, task-specific dataset. 

{% include figure.html path="assets/img/2023-02-07-taking_notes_on_the_fly/unsupervised.png" class="img-fluid" %}
<div class="caption">
    During unsupervised pre-training, the model is trained on a large unlabeled dataset. Then, it becomes a powerful baseline model that can be fine-tuned to work with various tasks. 
</div>
Generalization can be achieved with a sufficiently large model that is trained on sufficiently diverse and large data <d-cite key="radford2019language"></d-cite> <d-cite key="tirumala2022memorization"></d-cite>. However, pre-training large models is very time-consuming and costly in terms of environmental impacts and monetary resources <d-cite key="strubell2019energy"></d-cite> <d-cite key="chen2021bert2bert"></d-cite>. Thus, reducing the pre-training time and cost for transformer-based models is an imminent concern for machine learning practitioners. One area that has room for improvement is how quickly the model learns the embeddings of the rare words. It has been shown by many works that the embeddings of those words are noisy and not optimized <d-cite key="bahdanau2017learning"></d-cite>, <d-cite key="gong2018frage"></d-cite>, <d-cite key="khassanov2019constrained"></d-cite>, <d-cite key="schick2020s"></d-cite>. Furthermore, Wu et al. 2021 empirically observe that 20% of all sentences in the corpus contain a rare word and they propose a <em>"note-taking"</em> approach improves model’s ability to learn the embeddings of rare words <d-cite key="wu2021taking"></d-cite> . Impressively, they reduce the pre-training time of well-known large language models (LLMs), such as BERT, by 60%. The approach is called Taking Notes on the Fly (TNF) and we will dive deep into how it works in this blog post!

## Background
### Transformers

Wu et al. <d-cite key="wu2021taking"></d-cite>  extends the BERT model <d-cite key="devlin2018bert"></d-cite>, which is a transformer-based model, with an external memory. A transformer is composed of alternating multi-head attention and feed-forward layers. The initial input to the multi-head attention layer is the sum of word embeddings and positional embeddings. Each one-hot encoded token is multiplied with a weight matrix in order to obtain a real-valued non-sparse representation. The weight matrix is learned throughout the training. Because transformers do not process words in order, we also need to provide some information about the position of the token in a sentence. This is incorporated into the training by the "positional embedding (encoding)" $$(PE)$$ vector, composed of sine and cosine pairs.

$$PE_{\text{pos},2i} = sin(pos / 10000^{2i/d_{embed}} )$$

$$PE_{\text{pos},2i+1} = cos(pos / 10000^{2i/d_{embed}} ),$$

where $$pos$$ is the position of the token in the sentence, $$d_{embed}$$ is the embedding dimension of the model, and $$i$$ refers to the dimension in the $$PE$$ vector. Note that the positional embeddings do not depend on the meaning of the words, but only the position of them!

Self attention mechanism allows the model to relate words in a sentence through a set of learnable query $$(Q)$$, key $$(K)$$ and value $$(V)$$ vectors. The output of the attention function calculates a compatibility score for each pair of words in the sentence. Mathematically, self attention can be expressed as

$$ \text{self-attention} (Q,K,V) = softmax(QK^T / \sqrt{(d_k)}),$$ 

where $$d_k$$ is the dimension of hidden representations. 
In order to improve the representational power of the model, <d-cite key="vaswani2017attention"></d-cite> proposed a multi-head attention mechanism. In particular, the $$self-attention$$ function is calculated several times independently, results are concatenated, and linearly projected into the desired dimension. 

BERT is a masked language model which uses the transformer architecture. During training time, 15% of the words in the sentence are masked or replaced with a random word. The model learns to predict the words that are masked.

### Word Distribution in Texts

The distribution of the words in a natural language corpora follow Zipf's law <d-cite key="zipf1932selected"></d-cite>, that is, the frequency $$n^{th}$$ most frequent word is proportiional to $$1/n^\alpha, \: where \:\: \alpha \sim 1$$. 

{% include figure.html path="assets/img/2023-02-07-taking_notes_on_the_fly/zipf_law.png" class="img-fluid" %}
<div class="caption">
The frequencies of 50 most common words in Brown Corpus<d-footnote>Details can be found <a href="http://korpus.uib.no/icame/manuals/BROWN/INDEX.HTM"> in the link.</a></d-footnote>. Green is the word counts estimated by Zipf's law, blue is the actual count. Image is taken from <d-cite key="zipf"></d-cite>.
</div>
In other words, number of popular words are much less than of rare words, yet their frequency is much larger. This harms pretraining of LLMs because of the sparse and inaccurate optimization of neural networks, rare words are much likely to generate noisy and low-quality embeddings <d-cite key="gao2019representation"></d-cite>. 


## Related Work 

Pre-training of LLMs has become a burden in terms of training time and power consumption. Still, it is essential for almost every downstream task in NLP. This computational cost is addressed by several studies in terms of altering the model or utilizing the weight distribution of neural networks' layers. Particularly, <d-cite key="clark2020electra"></d-cite> added a discriminator to predict if each word in the sentence that is completed by the generator is correct or not. Another important work arised after the observation that the attention distributions of top and bottom layers are quite similar. <d-cite key="gong2019efficient"></d-cite> proposed an iterative algorithm that doubles the number of layers after each training episode. 

The efficiency of pretraining LLMs has shown to be incresed, still the heavy-tailed distribution of words in natual language corpora is an obstacle in further development <d-cite key="strubell2019energy"></d-cite>. 
Note taking approach positively impacts learning performance in humans <d-cite key="makany2009optimising"></d-cite>. This idea is inspired studies in terms of contributing to training efficiency <d-cite key="wu2021taking"></d-cite>  and increasing performance in downstream tasks <d-cite key="feng2022memory"></d-cite>, <d-cite key="fevry2020entities"></d-cite>, <d-cite key="guu2020retrieval"></d-cite>, <d-cite key="khandelwal2019generalization"></d-cite>. 

It is shown that the frequency of words affect the embeddings. Additionally, most of the rare words' embeddings are close to each other in embedding space indepent from its semantic information while the neighbors of frequent words are the ones that have similar meaning <d-cite key="gong2018frage"></d-cite>. Initial studies mainly used subword information to encode semantic information, this approach is shown to be valuable for morphologically rich languages <d-cite key="pmlr-v32-santos14"></d-cite>, <d-cite key="kim2016character"></d-cite>, <d-cite key="el2019parsimonious"></d-cite>. 
Recently, this problem is also adressed by using adverserial training where a discriminator classifies each word as 'frequent' or 'rare' allowing semantic information to be encoded  <d-cite key="gong2018frage"></d-cite>.

## Methodology

Because learning the embeddings of rare words is arduous, it takes a lot of training epochs for the model to make up for the resulting loss in quality. Thus, the authors propose keeping a third type of embedding (besides the word embeddings and positional embeddings), which is designed to retain additional information about the rare words. This embedding type can be considered as <em>taking notes</em> on the contextual information of these rare words as the training progresses, is also called the note dictionary, and is updated as the training progresses.

{% include figure.html path="assets/img/2023-02-07-taking_notes_on_the_fly/overview.png" class="img-fluid" %}
<div class="caption">
Figure 2 from the paper, which gives an overview of the note taking process. The note embeddings are randomly initialized and the other two embeddings are computed. Then, their sum is given to the transformer encoder as input. For every rare word encountered when going through the training data, its contextual information is calculated and the corresponding note embeddings are updated accordingly. This process goes on as the data is being fed to the transformer.
</div>
At this point, we assume that the text has already been pre-processed using Byte Pair Encoding (BPE<d-footnote>A very nice <a href="https://towardsdatascience.com/byte-pair-encoding-the-dark-horse-of-modern-nlp-eb36c7df4f10"> blog post</a> about BPE. Reading it is highly encouraged as it also provides visuals on BPE.</d-footnote>), which is a popular method that is used as a part of the text embedding process for NLP tasks <d-cite key="sennrich2015neural"></d-cite>. In BPE, each word is represented as a concatenation of sub-word units, which are selected according to how much each they unit occur in the given text. For example, if the sub-word <b>"pre"</b> occurs in the text frequently, it will be represented with a single character, such as <b>"X"</b> in this encoding. This way, the textual data is compressed and manageable. Also, because each sub-word unit gets their own embedding, we get a hybrid approach between word-level and character-level embeddings. Therefore, the embedding of each word might very well be made up of multiple consecutive tokens. With this information in mind, let us walk through the steps of note taking!

The first three steps are about initializing the required variables and determining the hyper-parameters of the scheme.

0a. Randomly initialize the note dictionary, $$NoteDict$$.

0b. Determine a window size ($$2k$$ as denoted in the paper), which corresponds to the number of surrounding tokens whose embedding will be included in the note.

0c. Determine a discount factor, $$\gamma\in (0,1)$$. This will determine 
how much weight we give to each occurrence of the rare word and the corresponding contextual information.

Now, note taking begins!

1.For each word $$w$$ in the training corpora, check if the word is a rare word or not. If it is rare, mark the index of the starting and ending sub-word tokens of the word with $$s$$ and $$t$$, respectively.

2.Compute the output of the transformer encoder on the input embeddings (positional+token+note embeddings). The output will be composed of $$d$$-dimensional vector per token. Call the output of the transformer
encoder on position $$j$$, $$c_j\in \mathbb{R}^d$$.

3.Given a sequence of tokens $$x$$ with word $$w$$ in it, sum the $$d$$-dimensional input embedding vectors of all tokens located between indices $$s-k$$ and $$t+k$$ and divide this sum by $$2k+t-s$$, namely, the number of tokens within that interval. The resulting vector is the note of $$w$$ taken for sequence $$x$$, $$Note(w,x)$$. Mathematically, we have
$$Note(w,x)=\dfrac{1}{2k+t-s}\sum_{j=s-k}^{t+k}c_j$$.

{% include figure.html path="assets/img/2023-02-07-taking_notes_on_the_fly/numberline.png" class="img-fluid" %}
<div class="caption">
    This figure demonstrates contextual embedding vectors at which locations will be selected and summed with an example. This line represents the indices of a sequence of length 11. Let us assume that the rare word is contained within tokens 4 to 6, and k=2, which makes the window size 2k=4. Thus, we sum the tokens at location 4, 5, 6, as well as 3, 4, (which are the two immediate left tokens) and 7,8 (which are the two immediate right tokens). Finally, we divide the each element of the resulting vector by 6, which is the total number of elements in the interval.
</div>

4.To update the note embedding of w, NoteDict(w), take the exponential moving average of its previous value and Note(w,x) using the discount factor, namely, 
$$NoteDict(w)=(1-\gamma)NoteDict(w)+\gamma Note(w,x)$$. This way, we can choose how much importance we assign to each occurrence of a rare word.

This process repeats until all of the sentences are processed this way. Note that, this can be achieved on the fly, as the model processes each sentence. Now that we have our notes neatly stored in $$NoteDict$$, let us incorporate them into the training process! We again take the exponential moving average of the sum of the positional and token embeddings (the embedding used in the original transformer paper) with the corresponding $$NoteDict$$ value using another parameter called $$\lambda\in(0,1)$$. In particular, for every word $$w$$ that occurs in both $$NoteDict$$ and sequence $$x$$, each location corresponding to the word $$w$$ and its surrounding $$2k$$ tokens is set to the weighted of the sum of the positional and token embeddings with the corresponding NoteDict value. Any other location is set to the sum of the token embeddings and positional embeddings only. The resulting vector will be the input to our model for the next step. Mathematically, for location $$i\in[d]$$, which corresponds to (one of the) tokens of word $$w$$ in the sequence, we have 
$$
\text{input}_i= \begin{cases} 
      (1-\lambda)(\text{p_embed}_i+\text{t_embed}_i)+\lambda\text{NoteDict}(w), & \text{w is a rare word}  \\
      \text{p_embed}_i+\text{t_embed}_i, &\text{otherwise} \\
   \end{cases}
$$
where $$\text{p_embed}$$ is positional embeddings, $$\text{t_embed}$$ is token embeddings and $$\lambda$$ (set to 0.5) is the hyperparameter specifying the weight of the notes when computing the embeddings.
## Results
{% include figure.html path="assets/img/2023-02-07-taking_notes_on_the_fly/graphs.png" class="img-fluid" %}
<div class="caption">
    Figure 3. from the paper, presenting the loss and GLUE scores of the models with and without taking notes, over many iterations.
</div>
The experiments are conducted on BERT and ELECTRA models. The loss values of the pre-training runs with <em>note taking</em> descrease significantly faster than vanilla pre-training. Moreover, the models trained while taking notes achieve higher GLUE <d-cite key="wang2018glue"></d-cite> scores much faster. Additionally, they report that after one million iterations, the GLUE score of the models pre-trained with notes are superior to their counterparts trained without notes. Finally, they report that when it took one model with note taking to reach a certain GLUE score around 100.000 training iterations, it took the model around 400.000 training iterations to reach that same score without notes. That is a 60% improvement in training time to reach the same performance! 


## Conclusion
The ever-increasing data sizes, enlarging models, and hardware resources are some of the major factors in the current success of LLMs. However, this also means immense power consumption and carbon emission. Because pre-training of LLMs is the most computationally intensive phase of a natural language task, efficient pre-training is the concern of this paper. Knowing that the heavy-tailed distribution of word frequencies in any natural language corpora may hinder the efficiency of pre-training, improving data utilization is crucial. Therefore, the authors propose a memory extension to the transformer architecture: "Taking Notes on the Fly". TNF holds a dictionary where each key is a rare word. The values are the historical contextual information which is updated at each time the corresponding word is encountered. The dictionary is removed from the model during the inference phase. TNF reduces the training time by 60% without any reduction in the performance.

