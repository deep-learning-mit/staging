---
layout: distill
title: Transformer Based Chess Rating Prediction
description: Your blog post's abstract.
  This is an example of a distill-style blog post and the main elements it supports.
date: 2023-11-10
htmlwidgets: true

# Anonymize when submitting
# authors:
#   - name: Anonymous

authors:
  - name: Anonymous

# must be the exact same name as your blogpost
bibliography: 2023-11-10-transformer-elo-prediction.bib

# Add a table of contents to your post.
#   - make sure that TOC names match the actual section names
#     for hyperlinks within the post to work correctly.
toc:
  - name: Proposal
    subsections:
      - name: Data
      - name: Methods
      - name: Evaluation
      - name: Relation to Course Material
# Below is an example of injecting additional post-specific styles.
# This is used in the 'Layouts' section of this post.
# If you use this post as a template, delete this _styles block.
# _styles: >
#   .fake-img {
#     background: #bbb;
#     border: 1px solid rgba(0, 0, 0, 0.1);
#     box-shadow: 0 0px 4px rgba(0, 0, 0, 0.1);
#     margin-bottom: 12px;
#   }
#   .fake-img p {
#     font-family: monospace;
#     color: white;
#     text-align: left;
#     margin: 12px 0;
#     text-align: center;
#     font-size: 16px;
#   }
---

## Proposal

Motivated by a lack of concrete methods to estimate an unrated or unknown chess player’s skill, we present Transformer-Based Chess Rating Predictions. Our main idea is to train a transformer based architecture to predict the elo rating of chess players from the sequence of moves they make in a game.

### Data

We can get data for games [here](https://database.lichess.org/#standard_games). For each game, we can consider the average rating of the players to be the thing we are trying to predict (we will only take games where players are within 400 rating points of each other). We may relax this restriction later on to include games with any rating gap, but we foresee difficulties in trying to disentangle the individual ratings in a given game. Our architecture is more suited to predicting the average rating between the two players, and the effect of differing playing styles may inject extra variance into rating predictions of individuals. We would be open to suggestions on how we could remedy this issue.

### Methods

One key decision we will have to make is on the best way to represent the data. Our current idea is to represent the game as a sequence of 3D Tensors, where each 2D “slice” represents some feature of the game state (positions of white pawns, castling rights, move repetitions, etc.). Crucially, we’ll also include the last move’s centipawn loss, which is a nonnegative measure of accuracy calculated by subtracting the engine evaluation of the played move from the engine evaluation of the engine-recommended move. Hopefully, this somewhat noisy notion of accuracy along with the context of the game state will provide enough information for the model to make accurate predictions.

Our main architecture will consist of a transformer with an autoregressive attention mask. Each game state is fed through an initial linear layer to generate initial embeddings, after which they’re inputted into a transformer in which a token only attends on itself and tokens that come before it. The final layer consists of a linear layer that maps to a final rating prediction, which we will evaluate with MSE.

### Evaluation

To see if our transformer model is truly learning anything from the game states, we can compare our transformer-based model with a simpler baseline model: for example, an LSTM that predicts the same average rating where the only inputs are the moves’ centipawn losses. We would like our transformer’s MSE to be significantly lower than the LSTM’s MSE over our testing dataset.

It would also be interesting to examine model behavior on “atypical” data - for example, on games with large rating gaps between two players or on tactically complex games in which even master-level players would make ample mistakes.

### Relation to Course Material

Our goal for this project is to improve our understanding of how to apply the more abstract concepts around transformers and input representation that we learned in class to a more concrete problem, and gain insight on what matters when optimizing the accuracy of our model (width vs depth of model, amount of data, diversity of data, amount of time to train, etc). Although we know the concepts behind what “should” improve accuracy, it would be interesting to see it play out in and the relative importance of different concepts (ex: perhaps, having a deeper model is not nearly as important as training for a long time).

https://arxiv.org/pdf/1908.06660.pdf (can use a similar board representation)
