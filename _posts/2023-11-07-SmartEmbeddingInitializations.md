---
layout: distill
title: Injecting Node Information via Embedding Initializations
description: Your blog post's abstract.
  This is an example of a distill-style blog post and the main elements it supports.
date: 2023-11-07
htmlwidgets: true

# Anonymize when submitting
# authors:
#   - name: Anonymous

authors:
  - name: Emma Tysinger
    url: "https://www.linkedin.com/in/emma-tysinger/"
    affiliations:
      name: MIT
  - name: Sam Costa
    url: "https://www.linkedin.com/in/samuelcos/"
    affiliations:
      name: MIT

# must be the exact same name as your blogpost
bibliography: 2023-11-07-SmartEmbeddingInitializations.bib 

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

This project will take a deeper dive into node embedding initializations for graph neural networks. We will explore the question can additional node information be injected into the model by using intentional embedding initializations rather than random initializations? Furthermore, are the learned embeddings better representations of the nodes? 

Graph Neural Networks (GNNs) have emerged as a transformative tool in machine learning, with the ability to capture the complex structures and relationships inherent in data. In molecular property prediction, for example, GNNs are great at encoding the atomic structure and intermolecular forces into high-dimensional embeddings, leading to more accurate predictions of chemical properties and drug efficacy. GNNs have also be used in traffic time prediction problems, physic simulations and social media analysis applications. Through message-passing and updating, GNNs are capable of learning embeddings that encode informations of node neighbors and long-distance complex connections – that we, as humans, may not be able to make. The quality of the embeddings is not only important for the accuracy of the task the GNN is trained on, but quality node embeddings can be used through transfer learning – enabling models trained on one task to adapt and excel in another. The importance of good embeddings in GNNs is why we want to look closer at embedding initializations and if we can inject additional information – not present through in the graph – to result in better learned embeddings after training.

Possible applications of initial embedding initializations could help in the field of drug discovery. For GNNs used for protein retrieval trained on a biomedical knowledge graph, using ESM embeddings for the proteins could add structure information that is not previously encoded in the graph entities. 

Our project will consist of two parts. We will use a GNN, TxGNN, that is implemented for disease-drug link prediction on a biomedical knowledge graph as a baseline model. The first part will be focused on modifying the GNN for protein-molecular function, retrieving the embeddings and training. We will train two models, one with random initializations and a other with embeddings initialized as ESM embeddings for the protein nodes. 
The second part of the project will focus on evaluating our models.

#### Embedding Quality Analysis
- Assess the qualitative differences in embeddings between random and intentional initializations.
- Perform intrinsic evaluation by measuring how well the embeddings capture semantic similarity or relatedness.
- Question to consider: Does embedding quality improve using intentional initializations, that could be used for downstream tasks via transfer learning?

#### Node Clustering
- Visualization of node embedding latent space using t-SNE plots and heatmaps
- Question to consider: Do the optimized model embeddings maintain information injected from the non-random initializations? Or do embeddings from both models converge to similar optimal embeddings?

#### Link Prediction Accuracy
- Determine if embeddings initialized with additional node information improve the performance of link prediction tasks compared to randomly initialized embeddings

## Introduction
Graph Neural Networks (GNNs) have emerged as a transformative tool in machine learning, with the ability to capture the complex structures and relationships inherent in data. In molecular property prediction, for example, GNNs are great at encoding the atomic structure and intermolecular forces into high-dimensional embeddings, leading to more accurate predictions of chemical properties and drug efficacy. GNNs have also be used in traffic time prediction problems, physic simulations and social media analysis applications. Through message-passing and updating, GNNs are capable of learning embeddings that encode informations of node neighbors and long-distance complex connections – that we, as humans, may not be able to make. The quality of the embeddings is not only important for the accuracy of the task the GNN is trained on, but quality node embeddings can be used through transfer learning – enabling models trained on one task to adapt and excel in another. The importance of good embeddings in GNNs is why we want to look closer at embedding initializations and if we can inject additional information – not present through in the graph – to result in better learned embeddings after training.

Possible applications of initial embedding initializations could help in the field of drug discovery. For GNNs used for protein retrieval trained on a biomedical knowledge graph, using ESM embeddings for the proteins could add structure information that is not previously encoded in the graph entities. 

### Project Outline
We will explore the question can additional node information be injected into the model by using intentional embedding initializations rather than random initializations? Furthermore, are the learned embeddings better representations of the nodes? To answer this question we will follow the steps outlined below:

1. We will download a precision medicine knowledge graph that and use a GNN, TxGNN, that is implemented for disease-drug link prediction on a biomedical knowledge graph as a baseline model. 
2. We will modify the GNN for protein-molecular function link prediction.
3. Generate and download ESM embeddings for each protein  
4. Pretrain and finetune two models – one using random protein node initialization and one using ESM embeddings for protein node initialization. We must pretrain our own models, rather than use the already pretrained model, since we are focusing on how different node initializations impact the predictive power. 
5. Evaluate both models 
6. Visualize latent spaces before pretrain, after pretraining and after finetuning

## Data
We used a precision medicine knowledge graph (PrimeKG) constructed by Marinka Zitnik's group at Harvard. PrimeKG compiles data from knowledge bases that coverage a broad variety of biomedical information including human disease, drug-protein interactions, genes and proteins with their associated biological processes, functions and cellular component, etc. PrimeKG contains 10 different node types – shown above – and 29 different types of undirected edges. There are over 120,000 nodes in total and over 8 million edges. What PrimeKG lacks, importantly, is any nodes or encodings of structural, molecular or sequenctial information for entity nodes such as proteins and drugs. 

The other data used were ESM embeddings for proteins in PrimeKG. The reason we were interested in using ESM embeddings, rather than embeddings from other protein foundation models, was that structural information was not already captured in PrimeKG, as previously mentioned. To obtain the ESM embeddings, first we downloaded the amino acid sequence for each protein from NCBI using Entrez. Then, using these sequences as input to Facebook's ESM2 model, we extracted the corresponding embedding.   

## Graph Neural Network Architecture

## Model Evaluation

## Latent Space Visualizations

## Discussion


  


