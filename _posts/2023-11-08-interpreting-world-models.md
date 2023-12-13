---
layout: distill
title: Interpreting decision transformers - world models and feature
date: 2023-11-08
htmlwidgets: true
authors:
  - name: Uzay Girit
    url: https://uzpg.me
    affiliations:
      name: MIT
  - name: Tara Rezaei
    affiliations:
      name: MIT
---


### Goal of the project:
Decision transformers allow us to bypass the need to assign long term credits and rewards as well as make use of the existing transformer frameworks, bridging the gap between agents and unsupervised learning. Getting trajectories from a trained RL agent, we can then use LLM interpretability techniques to understand these models and how they solve decision making problems. This is more and more crucial as large transformer models become capable of more complicated tasks and are used as decision making agents.

### Potential Questions to answer

- How do deep learning agents/DTs form world models and how can we interpret those abstractions?
- How do DTs simulate agents to match different levels of performance/different objectives?
- What patterns can we notice here across tasks and what does this tell us about DNN agents?
- How are these representations used by the model to complete the task?
- How do they compare to RL agents in terms of performance, training, compute etc.
- How much can patterns and dynamics in the agents we interpret tell us about larger models and language modeling?

### Potential experiments and analysis
- run a sparse autoencoder on a decision transformer on different tasks
- see what what representational patterns we see across tasks
- analyze through ablations and explore how the model is influenced by the Reward To Go token
- look at attention patterns and how they relate to the action space

### Uncertainties
- In practice, how tractable will the interpretation of world representations be in the framework of sequence modeling?
- Should we approach this in the frame of transformers for sequence modeling or explore latent world representations like the *World Models* paper? Maybe the two can be combined?
- Is it useful to see how different encodings of the data induce different strategies?
- Is it feasble to aim for automating any part of the pipeline like feature labeling with GPT4, etc
### Related work:

- [Decision Transformers](https://arxiv.org/abs/2106.01345)
- [World Models](https://worldmodels.github.io/)
- [Emergent world representations](https://arxiv.org/abs/2210.13382)
- [Anthropic sparse auto-encoders for LLM interpretability](https://transformer-circuits.pub/2023/monosemantic-features)
- [Decision Transformers interpretablity](https://www.lesswrong.com/posts/bBuBDJBYHt39Q5zZy/decision-transformer-interpretability)


