---
layout: distill
title: Quantum Circuit Optimization wtih Graph Neural Nets
description: We propose a systematic study of architectural choices of graph nerual net-based reinforcement learning agents for quantum circuit optimization.
date: 2023-11-08
htmlwidgets: true

# Anonymize when submitting
# authors:
#   - name: Anonymous

authors:
  - name: Julian Yocum
    # url: "https://en.wikipedia.org/wiki/Albert_Einstein"
    affiliations:
      name: MIT
  # - name: Boris Podolsky
  #   url: "https://en.wikipedia.org/wiki/Boris_Podolsky"
  #   affiliations:
  #     name: IAS, Princeton
  # - name: Nathan Rosen
  #   url: "https://en.wikipedia.org/wiki/Nathan_Rosen"
  #   affiliations:
  #     name: IAS, Princeton

# must be the exact same name as your blogpost
bibliography: 2023-11-09-quantum-gnn.bib  

# Add a table of contents to your post.
#   - make sure that TOC names match the actual section names
#     for hyperlinks within the post to work correctly.
toc:
  - name: Proposal

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

## Proposal

One of the most notable technological developments of the past century has been computing based on binary bits (0’s and 1’s). Over the past decades, however, a new approach based on the principles of quantum mechanics threatens to usurp the reigning champion. Basing the informational unit on the quantum bit, or qubit, instead of the binary bit of “classical” computing, quantum computing takes advantage of the strange phenomena of modern physics like superposition, entanglement, quantum tunneling. 

Leveraging these as algorithmic tools, surprising new algorithms may be created. Shor’s algorithm, based on quantum algorithms, can solve classically hard cryptographic puzzles, threatening the security of current cryptographic protocols. Additionally, quantum computers can significantly accelerate drug discovery and materials science through quantum molecular dynamics simulations. They also show great potential in Quantum Machine Learning (QML), enhancing data analysis and pattern recognition tasks that are computationally intensive for classical computers.

Similar to classical computers, which base their algorithms on circuits, quantum computers build their quantum algorithms on quantum circuits. However, quantum computers are still in development and are incredibly noisy. The complexity of a quantum circuit increases its susceptibility to errors. Therefore, optimizing quantum circuits to their smallest equivalent form is a crucial approach to minimize unnecessary complexity. This optimization is framed as a reinforcement learning problem, where agent actions are circuit transformations, allowing the training of RL agents to perform Quantum Circuit Optimization (QCO). Previous techniques in this domain have employed agents based on convolutional neural networks (CNN) <d-cite key="fosel2021"></d-cite>.

My previous research has demonstrated that the inherent graphical structure of circuits make QCO based on graph neural networks (GNN) more promising than CNNs. GNNs are particularly effective for data with a graph-like structure, such as social networks, subways, and molecules. Their unique property is that the model's structure mirrors the data's structure, which they operate over. This adaptability sets GNNs apart from other machine learning models, like CNNs or transformers, which can actually be reduced to GNNs. This alignment makes GNNs a highly promising approach for optimizing quantum circuits, potentially leading to more efficient and error-resistant quantum computing algorithms.

The aim of this project is to systematically investigate the impact of various architectural choices on the performance of GNNs in quantum circuit optimization. This will be achieved through a series of experiments focusing on key variables such as the number of layers in the GNN, the implementation of positional encoding, and the types of GNN layers used.

Specific objectives include:

1. **Evaluating the Number of GNN Layers**: Investigating how the depth of GNNs influences the accuracy and efficiency of quantum circuit optimization. This involves comparing shallow networks against deeper configurations to understand the trade-offs between complexity and performance.
2. **Exploring Positional Encoding Techniques**: Positional encoding plays a crucial role in GNNs by providing information about the structure and position of nodes within a graph. This project will experiment with various encoding methods to determine their impact on the accuracy of quantum circuit optimization.
3. **Assessing Different Types of GNN Layers**: There are multiple GNN layer types, each with unique characteristics and computational models. This project aims to compare the effectiveness of different layer types, such as Graph Convolutional Networks (GCN), Graph Attention Networks (GAT), Residual Gated Graph ConvNets (ResGatedGCN), and others in the context of QCO.
4. **Benchmarking Against Existing Approaches**: The project will also include comprehensive benchmarking against existing QCO techniques, such as those based on CNNs, to quantify the improvements offered by GNN-based approaches.