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
{% include figure.html path="assets/img/PrimeKG.png" %}
We used a precision medicine knowledge graph (PrimeKG) constructed by Marinka Zitnik's group at Harvard. PrimeKG compiles data from knowledge bases that coverage a broad variety of biomedical information including human disease, drug-protein interactions, genes and proteins with their associated biological processes, functions and cellular component, etc. PrimeKG contains 10 different node types – shown above – and 29 different types of undirected edges. There are over 120,000 nodes in total and over 8 million edges. What PrimeKG lacks, importantly, is any nodes or encodings of structural, molecular or sequenctial information for entity nodes such as proteins and drugs. 

The other data used were ESM embeddings for proteins in PrimeKG. The reason we were interested in using ESM embeddings, rather than embeddings from other protein foundation models, was that structural information was not already captured in PrimeKG, as previously mentioned. To obtain the ESM embeddings, first we downloaded the amino acid sequence for each protein from NCBI using Entrez. Then, using these sequences as input to Facebook's ESM2 model, we extracted the corresponding embedding.   

## GNN

## Model Evaluation
{% include figure.html path="assets/img/Figure2.png" %}

### Testing Varying Hidden and Output Layer Dimensions
{% include figure.html path="assets/img/Testing_output_dim.png" %}



## Latent Space Visualizations
We first were curious whether, after training our model, the final embeddings retained structural information about the proteins. To do this, we first clustered the original ESM embeddings using K-Means clustering. Next we visualized the embedding space of the original ESM embeddings, the final embeddings from the ESM model and the final embeddings from the random model using t-distributed Stochastic Neighbor Embedding (t-SNE) for dimensionality reduction. From the t-SNE plot of original ESM embeddings, we can clearly see the clusters from K-Means which serves as a verification of our clustering technique. 

Looking at the embedding space for the ESM and random models, colored by ESM clusters, ... 

Given that the final embedding space for the ESM model didn't seem to retain much of the information for the ESM embedding initialization, we were curious whether the ESM and random embeddings converged to a similar space. To test this theory, we clustered the final ESM model embeddings and subsequently visualized the final embeddings of the ESM and random models using t-SNE and colored by those clusters. 

If the two models converged to similar embedding spaces, we'd expect to see that clusters found in one embedding space would also be found in the other. This is the case, as seen in the two plots below ... 

### Testing Varying Hidden and Output Layer Dimensions
As mentioned earlier, we tested different dimensions for the hidden and output layers to see whether more and less output dimensions would retain the original ESM embedding information. 

### Clustering by molecular function labels
Because our model's task was to predict links between protein and molecular function nodes, we were curious to see if the final embeddings for the protein nodes would cluster well on the function labels. However, this wasn't as straight forward as having 1 molecular function label for each protein node, because each protein may be linked to multiple molecular functions. One protein may have multiple molecular function Gene Ontology (GO) annotations because the GO database uses a hierarchical system to categorize functions, where broader functions encompass more specific ones. A protein can be involved in several distinct biochemical activities, each represented by its own GO term, reflecting the diverse roles a protein can play in the cell. Instead of a single label, we extracted a molecular function profile, $v_i$, for each protein where $v_i[j] = 1$ if a link exists between protein $i$ and function $j$. We then had a sparse matrix, $V^{i \times j}$. Before clustering, we performed dimensionality reduction using truncated SVD which is optimal for sparse matrices. Finally, we performed K-Means clustering.

## Discussion


  


