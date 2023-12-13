---
layout: distill
title: Are Transformers Gamers?
description: 
  We aim to explore whether transformers can be applied to playing video games, and specifically want to explore what they learn to attend to. 
date: 2023-11-09
htmlwidgets: true

# Anonymize when submitting
# authors:
#   - name: Anonymous

authors:
  - name: Ethan Yang
    url: "https://www.ethany.dev"
    affiliations:
      name: MIT

# must be the exact same name as your blogpost
bibliography: 2023-11-09-transformers-as-gamers.bib

# Add a table of contents to your post.
#   - make sure that TOC names match the actual section names
#     for hyperlinks within the post to work correctly.
toc:
  - name: Introduction

---

## Introduction

Transformers have seen strong performance in NLP <d-cite key="vaswani2023attention"></d-cite> and in computer vision tasks <d-cite key="dosovitskiy2021image"></d-cite>.

Games require strong sequential decision making in order to succeed. Previous methods to play games such as Dota 2 have used LSTMs and reinforcement learning<d-cite key="dota2"></d-cite>. Transformers have also seen success on RL baselines such as Atari<d-cite key="chen2021decision"></d-cite>. 

To explore this question, we aim to train a network to play 1v1 [generals.io](https://generals.io), a real-time turn-based strategy game. In generals.io, two players with a general spawn on a board with mountains and cities. Initially, players have no knowledge of other parts of the board besides the tiles immediately surrounding their general. Armies are the main resource of the game, which generate slowly from ordinary tiles, but quickly from cities. Using armies, players compete to capture terrain and cities, which also grants further vision of the board. The goal of the game is for the player to use their army to capture the tile of their opponent's spawn point. 

A typical game state will look like the following:
<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/2023-11-09-transformers-as-gamers/generals_pomdp.png" class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/2023-11-09-transformers-as-gamers/generals.png" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
  The left image shows what the game looks like while playing. Red only is able to see tiles adjacent to it, and every other tile is covered in a fog of war. The right image lifts the fog of war, and shows where blue's general is located. 
</div>

The game can be represented as a POMDP. The underlying state, which is the state of the whole board, can only be observed at tiles that are adjacent to tiles claimed by the player. In addition, both the board state and action space are completely discrete. While the space of all possible actions throughout the game is large, only a small portion of actions is usually valid at a time: valid actions move army from a tile that is owned by the player. 

We note that generals.io has a modest daily player base, and has had attempts to implement bots to play against humans. Currently, no bots have been able to defeat top humans in play. The top bots, such as [this one](https://github.com/EklipZgit/generals-bot), are implemented using rule-based logic. Previous machine-learning based bots have attempted to use a CNN LSTM in the model architecture, such as [this one by Yilun Du](https://yilundu.github.io/2017/09/05/A3C-and-Policy-Bot-on-Generals.io.html). He separately evaluates a supervised learning approach, as well as a reinforcement learning approach. 

## Proposed Method

A wealth of data (over 500,000 games, each containing likely hundreds of state-action pairs) are available via human replays. 

The game state comes in the form of 15x15 to 20x20 boards. Each cell can have an arbitrary amount of army on it, and a few different special terrain features. On each turn, an action consists of selecting a user-controlled tile and a movement direction. Games can last many hundreds of turns. 

We want to answer a few questions:
1. How does the performance of CNN LSTM compare to using a transformer?
2. What properties do transformers learn when applied to sequential decision making in a game?
3. Can we learn good representations for quantities such as army counts on each tile? 

To approach this, we want to start by using supervised learning on state-action pairs from human games. We will compare the CNN LSTM approach and a transformer based approach. For the transformer, I'm not sure whether it makes sense to split into patches as ViTs do, as each tile in the game has a very distinct meaning. We can explore this and try it out. 

Experiments will also be done on the representation of the underlying state, as well as how we handle the very long history of states and actions that can accumulate during the game.

A stretch goal will be to investigate reinforcement learning in order to fine-tune the learned model. 