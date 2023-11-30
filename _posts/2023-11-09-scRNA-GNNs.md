---
layout: distill
title: 6.s898 Final Project Proposal 
description: Investigating the biological underpinnings of latent embeddings for scRNA-seq data.

date: 2023-11-09
htmlwidgets: true

# Anonymize when submitting
# authors:
#   - name: Anonymous

authors:
  - name: Stephanie Howe
    url: 
    affiliations:
      name: MIT CSAIL


# must be the exact same name as your blogpost
bibliography: 2022-12-01-distill-example.bib  

# Add a table of contents to your post.
#   - make sure that TOC names match the actual section names
#     for hyperlinks within the post to work correctly.
toc:
  - name: Background
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

## Background
I am conducting my M.Eng in a computational biology lab in CSAIL, specifically doing multiomic analysis on Alzheimer's Disease and Related Dementias (ADRD) data. Single cell data like this is extremely high dimensional, think about a dataset that is on the scale of 10s or 100s of thousands of cells, each with 10s of thousands of “features,” aka genes or chromatin regions. Because of this, lower dimensional representations of these cells and clusters amongst them are valuable to help simplify our view of the data and extract value. Moreover, in the context of cells labeled with biomarkers and varying neurodegenerative diseases, it is in our interest to explore cell to cell neighborhoods and relationships to see how they are similar within and between disease classes.

## Proposal
Since the idea of cell neighborhoods and clustering is so important, thinking of single cell datasets as a graph comes to mind. I propose investigating the ability of GNNs to represent high dimensional single cell data as a low dimensional embedding. In particular, the scGNN package was built to do this and uses the embeddings to create cell clusters and impute the single cell expression matrices. We can explore the effectiveness of deep learning on singel cell data in a few ways.
First, we can explore the accuracy of scGNN in clustering cell types by comparing the clustering with our already labeled data.
Moreover, it would be interesting to investigate which genes are contributing most to the latent space embeddings of our data. To do so, we can correlate the embedding dimensions with the original gene expression values to identify genes that have the most influence on each dimension of the embedding. This will help us understand how GNNs are creating these embeddings and if they make sense on a biological level. 
Lastly, there is room to tie the project back to ADRD diagnosis. We can analyze the results of scGNN on different diagnoses and how the embeddings might differ for each.

The scGNN package is published [here](https://www.nature.com/articles/s41467-021-22197-x).

