---
layout: distill
title: VGAE Clustering of the Fruit Fly Connectome
description: An exploration of how learned Variational Graph Auto-Encoder (VGAE) embeddings compare to 
    Spectral Embeddings to determine the function of neurons in the fruit fly brain.
date: 2023-11-09
htmlwidgets: true

# Anonymize when submitting
# authors:
#   - name: Anonymous

authors:
  - name: Max Filter
    affiliations: 
      name: MIT

# must be the exact same name as your blogpost
bibliography: 2023-11-09-deep-connectome-clustering.bib  

# Add a table of contents to your post.
#   - make sure that TOC names match the actual section names
#     for hyperlinks within the post to work correctly.
toc:
  - name: Connectomooes, and what they can teach us
  - name: Unsupervised graph representation learning
  - name: Proposed research questions and methods

# Below is an example of injecting additional post-specific styles.
# This is used in the 'Layouts' section of this post.
# If you use this post as a template, delete this _styles block.
---

## Connectomes, and what they can teach us

{% include figure.html path="assets/img/2023-11-09-deep-connectome-clustering/fruit-fly-connectome.png" class="img-fluid" %}
<div class="caption">
    The fruit fly connectome.<d-cite key="winding2023connectome"></d-cite>
</div>

Everything you've ever learned, every memory you have, and every behavior that defines you is stored somewhere in the neurons and synapses of your brain. The emerging field of connectomics seeks to build connectomes–or neuron graphs–that map the connections between all neurons in the brains of increasingly complex animals, with the goal of leveraging graph structure to gain insights into the functions of specific neurons, and eventually the behaviors that emerge from their interactions. This, as you can imagine, is quite a difficult task, but progress over the last few years has been promising.

Now, you might be asking yourself at this point, can you really predict the functions of neurons based on their neighbors in the connectome? A paper published by Yan et al. in 2017<d-cite key="yan2017network"></d-cite> asked this same question, searching for an answer in a roundworm (C. elegans) connectome. In their investigation, they discovered a neuron whose behavior had not been previously characterized, which they hypothesized was necessary for locomotion. They tested this hypothesis by ablating the neuron on a living C. elegans, and to the dismay of that poor roundworm, found that it was indeed necessary.

Although impressive, the C. elegans connectome has only ~300 neurons, compared with the ~100,000,000,000 in the human brain; however, this year (2023):

1. A paper by Winding et al.<d-cite key="winding2023connectome"></d-cite> has published the entire connectome of a fruit fly larvae, identifying 3016 neurons and their 548,000 synapses.
2. Google Research has announced an effort to map a mouse brain (~100,000,000 neurons)<d-cite key="januszewski2023google"></d-cite>

This is exciting because the fruit fly dataset presents an opportunity to identify more nuanced functions of neurons that may be present in more complex species like mice, but not in simpler species like the roundworm. This creates the requirement for algorithms that are **sufficiently expressive** and able to disentangle the similarities between neurons that appear different, but are functionally similar. 

Furthermore, current efforts to map connectomes of increasingly complex animals makes it desirable to have algorithms that are **able to scale** and handle that additional complexity, with the hopes of one day discovering the algorithms that give rise to consciousness. 

## Unsupervised graph representation learning

The problem of subdividing neurons in a connectome into types based on their synaptic connectivity is a problem of unsupervised graph representation learning, which seeks to find a low-dimensional embedding of nodes in a graph such that similar neurons are close together in the embedding space.

A common way to identify functional clusters of neurons is through the lens of homophily, meaning that neurons serve the same function if they are within the same densely connected cluster in the connectome; however, this fails to capture the likely case that neurons with similar low-level functions span across many regions of the brain<d-cite key="winding2023connectome"></d-cite>. 

Instead, a better approach might be to cluster neurons based on their structural equivalence, such that groups of neurons with similar subgraph structures are embedded similarly, regardless of their absolute location in the connectome. This is the approach taken by Winding et al.<d-cite key="winding2023connectome"></d-cite>, who "used graph spectral embedding to hierarchically cluster neurons based on synaptic connectivity into 93 neuron types". They found that even though they used only information about the graph structure to predict functions, neurons in the same clusters ended up sharing other similarities, including morphology and known function in some cases.

Spectral embedding is a popular and general machine learning approach that uses spectral decomposition to perform a nonlinear dimensionality reduction of a graph dataset, and works well in practice. Deep learning, however, appears to be particularly well suited to identifying better representations in the field of biology (e.g., AlphaFold2<d-cite key="jumper2021highly"></d-cite>), and deep learning methods do appear to be capable of creating embeddings that more effectively preserve the topology of nodes in graphs<d-cite key="zhu2023unsupervised"></d-cite><d-cite key="kipf2016variational"></d-cite>. 

{% include figure.html path="assets/img/2023-11-09-deep-connectome-clustering/vgae-embedding.png" class="img-fluid" %}
<div class="caption">
    Learned VGAE graph embedding for Cora citation network dataset.<d-cite key="kipf2016variational"></d-cite>
</div>

Thus, it stands to reason that deep learning might offer more insights into the functions of neurons in the fruit fly connectome, or at the very least, that exploring the differences between the spectral embedding found by Winding et al. and the embeddings discovered by deep learning methods might provide intuition as to how the methods differ on real datasets.

## Proposed research questions and methods

In this project, I would like to explore the differences between functional neuron clusters in the fruit fly connectome identified via spectral embedding by Winding et al. and deep learning. Specifically, I am interested in exploring how spectral embedding clusters differ from embeddings learned by Variational Graph Auto-Encooders (GVAE)<d-cite key="kipf2016variational"></d-cite>, which are a more recent architecture proposed by one of the co-authors of the Variational Auto-Encoders (VAE) paper<d-cite key="kingma2013auto"></d-cite>, Max Welling. I believe GVAEs are an interesting intersection of graph neural networks (GNNs) and VAEs, both of which we explored in class, and that comparing this technique to spectral embedding is also relevant to our learning, because spectral decomposition has been discussed in class with respect to network scalability and RNN weights. My hypothesis is that a deep learning technique would be better suited to learning graph embeddings of connectomes because they are able to incorporate additional information about neurons (such as the neurotransmitters released at synapses between neurons) and are able to learn a nonlinear embedding space that more accurately represents the topological structure of that particular connectome, learning to weight the connections between some neurons above others.

My proposed research questions that I'd like my project to address are:

- How do unsupervised deep learning approaches for clustering graph nodes based on structural similarity compare to more traditional machine learning approaches like spectral embedding?
- How does the theory of Graph Variational Autoencoders combine what we learned about VAEs and graph neural networks? Since both VAE and VGAE have the same co-author, I assume the theory is similar.
- Which methods are more efficient and would scale better to large datasets (e.g. the mouse connectome)?
- How do connectome clusters learned by GVAE compare to the spectral clusters found in the paper? 

My project would make use of the fruit fly connectome adjacency matrix provided by Winding et al. as its primary dataset.