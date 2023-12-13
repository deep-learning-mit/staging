---
title: Studying the benefits and limitations of sparse auto-encoders for compositional reasoning tasks
layout: distill
date: 2023-12-12
htmlwidgets: true

authors:
  - name: Uzay Girit
    affiliations:
      name: MIT
  - name: Tara Rezaei
    affiliations:
      name: MIT

bibliography: 2023-12-12-sparse-autoencoders-for-othello.md

toc:
  - name: Introduction
  - name: Background and related work
  - name: Method and setup
  - name: Results
    subsections:
      - name: Comparison to Pythia-70m dictionaries
      - name: Investigating the effect of size
      - name: Interpreting the sparse autoencoder
        subsections:
          - name: "H1: Location features"
          - name: "H2: Predictive features"
  - name: Discussion and Conclusion
---

# 6.S898 Project - Studying the benefits and limitations of sparse auto-encoders for compositional reasoning tasks

# Introduction

Neural networks accomplish complex tasks and are poised to be increasingly used in critical and ubiquitous sectors of civilization. But given a model seemingly solving a problem, how much can we say about precisely how it does that and what its solution looks like?

It might seem like this type of question would be hopeless, but interpretability has been progressing and we can make some headway on questions like these. One of the issues for interpretability is the fact that networks pack a lot of information into individual neurons in complex hard to separate ways, which means it's hard to look at top activating examples for a neuron and see what it's doing. This is [superposition](https://arxiv.org/abs/2209.10652). [Anthropic's recent paper](https://transformer-circuits.pub/2023/monosemantic-features/index.html) leveraged sparse autoencoders (*SAEs*) to learn an interpretable basis for LLM features. Sparse autoencoders are weak dictionary learning algorithms that leverage autoencoders trained to encode and then decode the activations of a certain module in the model. Contrary to classical auto-encoders, the hidden state does not necessarily have to be smaller (enforcing compression), but the mapping has to be sparse, which we enforce by penalizing the L1 norm of the activations, where L1 is just the sums of the absolute values. This makes the feature basis much more disentangled, clean and sparse.

That paper is far reaching in its results and suggests a lot of potential for SAE interpretability methods. However our work wants to investigate how effective SAEs are in contexts where there is a lot of compositional reasoning. Indeed, the a lot of the features they find hinge on the fact that their simple 1L language model is picking up on a lot of cleanly separatable cues and heuristics that are feeding into its prediction -- for example a feature that's high for arabic text, or in HTML contexts, etc.... But this seems like it'd be harder if we have a model composing reasoning and computation across steps in by nature entangled ways.

So we decided to see how this method would perform on a task where there are plausibly much less heuristic features that are are separable, and intuitively requires more compositionality and reasoning than the capabilities of a small 1 layer language model. We turned to the game of Othello, for which some ML interpretability has already been done, making our analysis easier, and applied sparse autoencoders to see how they would perform and what we could learn from them. We picked Othello because it's a complex task where it might seem intuitive that the model has to gradually compose information across layers and reason about what types of moves and positions might be valid. Indeed, in the original Othello-GPT paper, they find a linear world representation when you feed the model sequence data, suggesting complex reasoning patterns. This is an initial analysis and there are many things we'd be excited to see that would make this more fleshed out.

# Background and related work

**Sparse Autoencoders**: There is some previous work done on [dictionary learning](https://arxiv.org/abs/2103.15949) to interpret neural networks. The idea of sparse dictionary learning is to find an over-complete basis (ie there are more basis vectors than dimensions) in your embedding space, such that on inputs in your data most of the dictionary basises are orthogonal to your data, and only a few activate (sparsity). This has been used very recently to visualize transformer features for language models, as a way of taking internal feature representations out of [superposition](https://transformer-circuits.pub/2022/toy_model/index.html).Superposition is a barrier to interpertability where neurons and features are encoding a lot of things at once, making it hard to study on individual behaviors and parts of the model. Most recently, Anthropic did extensive interpretability work on a 1 layer transformer by using sparse autoencoders in [decomposing language models](https://transformer-circuits.pub/2023/monosemantic-features#related-work). They learned a sparse embedding space and then conducted a lot of analysis and interpretability on the features the original network was learning by studying it in the sparse embedding space.


**Transformers for reasoning tasks and Othello:**
Transformers and specificially [decision transformers](https://arxiv.org/pdf/2106.01345.pdf) have formerly been used for more complicated tasks than natural language sequence prediciton like reasoning tasks and games and proven to be successful. Although cutting edge LLMs exhibit strong reasoning capabilities, toy models and small languag models that are more accessible and that people are trying to use for interpretability are quite small, limiting their reasoning ability. Othello is a simple to understand but complex to win two player board game, where you gradually place pieces and try to "capture opponent" pieces by sandwiching rows, columns, and diagonals of the board with two of your pieces. The winner is the player with the most pieces at the end. [Recent work](https://arxiv.org/pdf/2210.13382.pdf) lead to the creation of a dataset of Othello games and the publishing of a model called Othello-GPT that learns to play Othello successfully. We use both of these in our work. The way they train the model is by giving it sequences of Othello moves from games, and asking it to predict the next move, in an unsupervised way, obtaining a model that can predict legal moves and understands the mechanism of the game. They show the existence of representations forming in the model, by using a probe to recover the full board state from the model activations, even though it's just given a sequence. This suggests the model learns more than just heuristics and is able to do internal reconstruction of the game's features.

**Interpreting features and circuits**
In the original Othello-GPT, their world model probe was nonlinear. Neel Nanda [extended their work](https://www.neelnanda.io/mechanistic-interpretability/othello) and found a linear world representation of the othello model, by seeing that instead of representing the state as "black's turn" vs "white's turn", the model represented it in an alternating manner, distinguishing between "my turn" vs "their turn".  There is also some other work on [interpreting](https://www.lesswrong.com/posts/bBuBDJBYHt39Q5zZy/decision-transformer-interpretability) transformer models outside of the context of language modeling, for example with decision transformers, but this is very much a growing subfield. We were also able to get a better intuition for the features in the othello model by using [neuron visualization data published by the authors](https://kran.ai/othelloscope/index.html).

# Method and setup

In order to investigate a reasoning task, we used a synthetic GPT model trained on a dataset of valid Othello game sequences of length 60 [(by Li et al)](https://github.com/likenneth/othello_world). We manipulate and access the model's activations and internals using the [TransformerLens](https://neelnanda-io.github.io/TransformerLens/) library.

We used the MSE loss as a baseline to compare the performance of sparse autoencoders on a reasoning tasks versus a natural language sequence prediction task. We replicated the training of a recent [set of dictionaries](https://www.alignmentforum.org/posts/AaoWLcmpY3LKvtdyq/some-open-source-dictionaries-and-dictionary-learning) of similar size on the GPT language model (EleutherAI's 6-layer pythia-70m-deduped) and compare our results.

Our set up for the replication, where we pick the same hyperparameters as the authors, consists of an 8-layer [GPT](https://openai.com/research/language-unsupervised) model  with an 8-head attention mechanism and a 512-dimensional hidden space. We set up a buffer that gathers the model's activations on a batch of game data and uses it to train the autoencoder. The buffer automatically runs the model on another batch of data once it is half empty. The activations then get fed into the autoencoder's training loop, where it optimizes to minimize the reconstruction loss of form $L = L_1 + L_2$. In this equation, $L_1$ is the term originating from the $L_1$ norm of the weights, with a sparsity coefficient of $1e-3$ for the encoder of size $16 \times  512 = 8192$ a sparsity coefficient of $3e-3$ for the size $64 \times 512 = 32768$ and $L_2$ is the term originating from the square error of the reconstruction with regards to the actual model investigations.

We then train various sizes of sparse autoencoders on the 4th layer of the othello model and investigate the impact of the autoencoders size on the reconstructed hidden state.

We measure the reconstruction power of the encoder with a reconstruction score defined as $\frac {Loss_{ZeroAblation} - Loss_{Reconstruction}} {Loss_{ZeroAblation} - Loss_{Normal}}$ where $Loss_{ZeroAblation}$ is Loss after ablating the reconstructed layer and use this as a measure for how well the encoder is able to reconstruct the mlp layer. The intuition behind this is that we compare a "base zero", which is the ablation loss, with both the reconstruction of the layer and the original construction of the layer. This will provide us with a metric of how close our reconstruction is to ground truth.



# Results


## Comparison to Pythia-70m dictionaries


The following tables are the results from training a sparse autoencoder of size $16 \times  512 = 8192$ and $L_1$ penalty coefficient of $1e-3$.

Encoder's Measured MSE loss on OthelloGPT after 100000 epochs.

| Layer |  MSE  |
|:-----:|:-----:|
|   0   | 0.370 |
|   1   | 0.537 |
|   2   | 0.686 |
|   3   | 0.833 |
|   4   | 0.744 |

Encoder's reported MSE loss on Pythia-70m after 100000 epochs.
| Layer |  MSE  |
|:-----:|:-----:|
|   0   | 0.056 |
|   1   | 0.089 |
|   2   | 0.108 |
|   3   | 0.135 |
|   4   | 0.148 |




The following tables are the results from training a sparse autoencoder of size $64 \times 512 = 32768$ and $L_1$ penalty coefficient of $3e-3$

Encoder's Measured MSE loss on OthelloGPT after 100000 epochs.

| Layer |  MSE  |
|:-----:|:-----:|
|   0   | 0.749 |
|   1   | 0.979 |
|   2   | 1.363 |
|   3   | 1.673 |
|   4   | 2.601 |

Encoder's reported MSE loss on Pythia-70m after 100000 epochs.
| Layer |  MSE  |
|:-----:|:-----:|
|   0   | 0.09  |
|   1   | 0.13  |
|   2   | 0.152 |
|   3   | 0.211 |
|   4   | 0.222 |


From the results above we can see that the autoencoder reconstructs with higher MSE loss despite having the same sparsity constraint and multiplier between the activation size and the sparse embedding. The difference becomes more drastic as we increas the sparsity of the encoder. Our analysis of these results is that this aligns with our hypothesis in natural language sequence prediction for small models like these, it might be that it is easier for the encoder to learn sparser and more easily separable features that allow it to recover the activations. However, on a task like playing the game of Othello where the features are more abstract, and we think there might be a higher requirement of complex compositionality across layers, increasing sparsity and size, makes the model perform worse.

Another significant emerging pattern in the MSE loss of the encoders is the fact that loss increases in the furthur layers, which backs up our initial claim; that as features become more abstract, the autoencoder has a harder time reconstructing them.

It is worth noting that the increase of MSE across the two sets of tables is impacted by both the increase in size and sparsity. We had made the two tables, to match the already existing [benchmarks](https://www.alignmentforum.org/posts/AaoWLcmpY3LKvtdyq/some-open-source-dictionaries-and-dictionary-learning). However, in the following, we include the results of a sparse autoencoder with penalty coefficient of $3e-3$ and size $16 \times  512 = 8192$ to validate our claims about sparsity, without the effect of size.

Encoder's Measured MSE loss on OthelloGPT after 100000 epochs.

| Layer |  MSE  |
|:-----:|:-----:|
|   0   | 0.749 |
|   1   | 0.979 |
|   2   | 1.363 |
|   3   | 1.673 |
|   4   | 3.105 |

## Investigating the effect of size

In furthur investigation, we experimented with training various sizes of autoencoders on layer 4 of the model. The size of the autoencoder is determined by the equation $size = x \times 512$ where $x$ is the size factor. We vary the size factor from $0.25$ to $32$. The size factor describes how much our autoencoder embedding space is bigger than the original activation space, therefore deciding how much "extra space" the autoencoder has to obey the sparsity constraint and preserve good reconstruction. We included smaller sizes so that we could investigate the effect of size and whether the encoder would be able to learn more compact features and still perform well. Our results are found in the following:

![recons_loss vs epochs](https://hackmd.io/_uploads/S1GB0NBUp.png)

As seen in the figure above, we see reconstruction loss decrease significantly as the number of dimensions in the autoencoder's hidden space becomes larger than the original space. A sparse autoencoder with less dimensions than the original latent space fails to reconstruct well and this can be even better observed in the following figure.

![Screenshot 2023-12-11 at 8.47.16 PM](https://hackmd.io/_uploads/BJAJerHLa.png)

This picture suggests that maybe if we scale up sparse auto encoder embedding size we can recover performance at low cost. However, Anthropic's interpretability work, linked earlier, suggests that as you increase the size of your autoencoder embeddding, you risk getting a lot of niche highly specific features with complex interactions, therefore making interpretability harder. For example, at a given size they observe a base64 feature that fires for base64 text, and then at a larger size they see it splits into several base64 features that activate for slightly different token beginnings.

These results highlight the challenge of sparse autoencoders for compositional tasks, and bring us to the question of interpreting sparse embedding spaces for compositonal reasoning.

## Interpreting the sparse autoencoder

Here we had to take a detective's approach and form different hypotheses of what the model was doing and how to test them. This analysis is exploratory, and given more time we'd be excited about extending this/doing even more experiments to get a complete picture. However, we're excited about what we found and are confident that this approach is promising.

We started by caching the autoencoder embeddings on a subset of data with valid Othello sequences and moves. This gave us a dataset to work with.

We then did some macro level analysis by looking at and inspecting random features (dimensions of the embeddings) and seeing what kinds of boards activated most on them (by activated most we mean that the feature had a high value on that input activation for that board). This somewhat followed the pattern laid out by [Anthropic's analysis](https://transformer-circuits.pub/2023/monosemantic-features/index.html#global-analysis-interp).

However, in Anthropic's 1L language model paper they have the following figure:

![image](https://hackmd.io/_uploads/SyIELvLIT.png)

They are indicating that in their setup most of the features seem to be interpretable and clear to a human, according to human scores. In our experience looking at our sparse autoencoder and top activating examples for different features, it seems that a lot of the features are still not interpretable and we will need more work to understand the full picture [^1]. This may be because a lot of semantic cues for simple language modeling are more aligned with our human understanding, in the sense that the concepts the model operates on are pretty intuitive, whereas for Othello it has to build a compositional model of the game state across layers, in ways that are potentially less likely to correlate with how we might perceive the problem. We don't claim that there are not such complex dynamics in even simple language models (there definitely are!), but we think there are more simple patterns to pick up on. We believe that the method laid out in that work needs to be extended to be applied to compositional networks for reasoning adjacent tasks, because it does not seem sufficient for this Othello model. This is an empirical claim based on studying and looking at a lot of data on when sparse features were activating throughout the Othello dataset.

To do some global analysis, we computed a frequency histogram of the values of each feature on the dataset, and then we took an average of this frequency histogram to get a full picture of how often and how strongly features are activating across the dataset. This is on a log scale.

![image](https://hackmd.io/_uploads/B1V7_HIL6.png)

As we can see, on average for each feature there are a lot of inputs where the feature is not reading much at all, which makes sense given the sparsity constraint. Then as the activation gets higher and higher the frequency of each bucket decreases.

If we increased the sparsity regularization even more we might see a sparser activation graph with more high activing frequency for large activations, but in a lot of classic encoders the distribution of embeddings tends to have a lot of smaller noise around zero, where here a lot of our values are actually very often split into either zero, or something significant.

We then proceed to making some hypotheses about how the model might be localizing computation about the game board throughout its features, and make some tests to see what might be going on.

### H1: Location features

Hypothesis: what if there are features that represent the location of the last move, and only activate when that last move is within some cluster of the board? This would align with earlier world model wor.

This would be an example of a strong monosemantic and interpretable feature.

However, we later realized that this is probably more likely as a more primitive pattern that would be noticed earlier in the model layers, before it then refines and comes up with information to decide what to predict.

Never the less, we looked at the contexts in which a feature is reading strongly, and thus found a list of high-activating moves for each feature (*for what current moves is feature j activating*). We then clustered these into 3x3 location clusters on the board, marking positions as the same if they were close in a small square. That was based on the idea that it does not have to be activating for the exact same current move but moves in general that are adjacent. These features would then represent: *was the current move around this position of the board?*.

This plot was computed by looking at those activating contexts for each feature and seeing how many non-adjacent clusters of positions are within those moves. We then compute a histogram on the cluster count, trying to see how many features activate locally in a small number of clusters.

![image](https://hackmd.io/_uploads/BymEFrU8T.png)

We can see that our hypothesis was wrong here and that at this point in the network our features are activating for current moves across the board, not really in a localized, and don't sparsely activate just when a given location is played. This was useful data to see and showed us that at this point in the network it was probably operating on high level features and things that could directly relate to its final prediction. The small amount of locally activating features all tend to just have small activations in general.

### H2: Predictive features

This brought us to the next experiment, where we wanted to test for higher level patterns related to its prediction.

We were curious studying the link between the times when a feature of our autoencoder is writing strongly on an input and the actual correct prediction for that input, ie the actual correct next token it's trying to predict. Is there a localization effect there where a feature activates highly only when the correct prediction is within some cluster?

We investigated and collected, for each feature, a list of the real (heldout) next action in the sequence whenever it is activating non negligibly. This gave us a sequence of next moves for each context where a feature wrote strongly to the activation output. Then we clustered these actions into regions of 3x3 squares on the board, trying to narrow in on the idea of local activation of a feature. We operationalized the notion of reading strongly on a game board by setting a threshold activation of 0.001 by looking at the earlier plot of activation distribution and seeing what made sense. This is actually pretty low, but it still stays significant because the sparsity constraint often just nulls out values when they are not relevant, so even low small values have signal.

This allows us to map each feature to a number of activating clusters.

We then plot a histogram for the number of clusters of next action locations for each feature in our dataset. The idea is that if a feature is activating on a small number of clusters for the next action, then it might be picking up in patterns on the board that are linked to the final model's prediction, in a consistent way based on the real result.

![image](https://hackmd.io/_uploads/Sy9PKBUIT.png)

It's interesting to compare this to the previous plot, as here there are actually a decent amount of features that seem localized, reacting and forming predictions based on what part of the board they think the next step or move might be in, and not activating across the board for the next token. These are the ~100s of features that are only reacting in some small amount of clusters, like two or 1.

It seems that in layer 4 in certain cases the model is already developing an idea of what the next move will be, and is localizing sparse features for different prediction areas.

This explanation is not explaining the full behavior and there is probably a lot going on to extend the prediction into higher layers. We can see this in the frequencies of all the features that are activating in a lot of different next-token contexts, probably picking up on general things on the board and harder to interpret compositional steps that will allow it to make predictions later.

This reminded us of the [logit lens] in language modeling where you can unembed the early activations and get coherent (and gradually improving as you increase the layer number) predictions for the next token. This seems to be showing that some of the features are already localizing predictions about the correct prediction, in a consistent manner.

We investigated those features corresponding to the left side of the plot ($1 \leq x \leq 3$, $x$ number of clusters) that activate only for some cluster of valid next sequence areas and found data that validated this impression! We hypothesize it's because some action predictions are pretty clear to predict early on based on good strategy and how the dataset of sequences was generated. We found features that consistently were activating for when a given board position was the correct next board position.

We focused particularly on feature #15 of our dim 4096 autoencoder, noticing through our analysis that it had interesting activation patterns.

We plotted its activation value histogram:

![image](https://hackmd.io/_uploads/Byk19HULT.png)

We can see a long sparse tail of inputs where the feature doesn't activate at all, and then a similar frequency for values beyond some threshold of activation.

On manual inspection, other than the big cluster of samples where it's reading zero or doesn't activate, the feature is basically always just activating when the next move is in a specific cluster at the bottom of the board. To be more precise, 90% of the boards where it activates with a value > 0.001 are in that cluster, 93% for 0.01,

Here are some of those example boards, where the next move played is G4, and the model activates strongly.

One of many examples of board where feature #15 activates strongly and in fact the next correct move is G4.
![image](https://hackmd.io/_uploads/BJZEDS8U6.png) [^2]

Example where the feature activates and the actual next move is F4, right above G4, in the same cluster:
![image](https://hackmd.io/_uploads/ryy8Jj8U6.png)

This is really interesting! Why does this feature exist? We've been thinking about the structure of Othello and the way the data was generated, and we think the idea is that the network is pretty confident about this position and early on manages to recognize and see what's going on with the rest of the board to put its hypothesis in this feature.

Although we haven't explained a lot of the other features, it's cool that this method has allowed us to understand and correlate this feature with a state of the game and the understanding the model has of the board!

# Discussion and Conclusion

We are excited about future work in this direction and think interpreting compositional computation circuits is key to understanding how tranformers and language models solve complex problems. In terms of our work with Othello GPT, we are excited about pushing sparse autoencoders further on this architecture and motivating more interpretability work. We are interested in work to train SAEs across layers and then see if we can track computation and model changes through sparse embeddings across layers, mirroring this [exploratory work]. This might be helpful to understand compositionality across layers. We also think interpreting features for SAEs with width smaller than the original width might be interesting to find projections of network activations that have very high level, compressed features, that might therefore be easier to interpret. We are also interested in methods that use SAE features to make causal statements aobut model behavior, for example by plugging the SAE into the model inference step, where at the end of our MLP we feed in the decoded encoded version of activations into the rest of the model. With this kind of setup you could then potentially ablate or modify different features to validate and study how your interpretability hypotheses about different parts of the model actually change its final predictions. Some of the limitations of our work is that we would have liked to run more experiments on different sparsity coefficients, and make more in depth comparisons to language models to see to what extent our arguments about compositional reasoning hold in a rigorous way. We would be excited to see how increasing sparsity even more affects our ability to interpret the model, potentially making things more tractable. We also recognize the difficulty of interpretability and have not been yet been able to interpret any of the more complex Othello SAE mechanisms.


To conclude, we've investigated the potential for sparse autoencoders for compositional reasoning tasks in the context of the Othello sequence prediction problem. Our hypothesis is that sparse autoencoders will be useful to understand such systems but their application will be more involved and complex than for earlier patterns found in language modeling tasks. We trained a sparse autoencoder at different layers of the network and see how its performance and capabilities differ compared to previous results on language. We observe our autoencoder trained with the same hyperparameters and scaling factor for size still struggles to reach the same reconstruction performance as those for language model activations. This reveals something about the structure of these data distributions, and supports our intuition that for simple small language models SAEs are particularly performant due to their ability to pick up on a lot of separable and sparse features, but for compositional solutions where the model is learning an algorithm across layers to solve a task, the sparsity constraint incurs more of a cost, which limits this method. This intuition stems from the idea that leveraging the full extent of neural activations for compositional tasks is key to build complex algorithms across layers, and maybe less so for prediction problems that are more tractable through the composition of independent heuristics. We also nonetheless do some interpretability on our trained autoencoder, and note that the features seem less directly interpretable than those for language model SAE features (as supported by our hypothesis), but that there is some signal to analyze and understand, giving us hope for future work to use SAEs to understand compositional reasoning and circuis in general. In particular, we look at the range and frequency of sparse activations, and form different hypotheses about the ways the model might be localizing computation in sparse embeddings. We find the existence of predictive neurons already at layer 4, that activate when the model is already confident about a specific next action to predict. Although much of the features remain obscure, our results indicate that although sparsity is a harder constraint to impose for compositional reasoning, it can still be a useful starting point to interpret model computation.

[^1]: To some extent increasing the sparse regularization penalty could help with this, but our exploratory analysis revealed that increasing the sparsity penalty made the model perform too badly on the data. We could always counter this by increasing the size of the encoder, but Anthropic's paper and our understanding suggests that this leads core interpretable features to split and split until it's hard to get a good picture of what's going on.

[^2]: these plots are both before the G4 cluster move is played.
