---
layout: distill
title: 6.s898 Final Project- Investigating the biological underpinnings of latent embeddings for scRNA-seq 
description: 

date: 2023-12-12
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
  - name: Background and Motivation
  - name: Graph Neural Networks (GNNs) as an architecture and their application to single-cell analysis
  - name: Intro to the Data
  - name: Applying scGNN to our AD scRNA-seq data
  - name: Visualizing the Degree Distribution of the Cell Graph
  - name: Understanding Cell Clusters in the Embedding Space
  - name: Exploring Alzheimer’s Related Gene Contributions to the Embedding Space
  - name: Wrapping it up
  - name: Future Analysis



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

## Background and Motivation
Neurodegenerative diseases represent a complex and diverse group of disorders characterized by the progressive degeneration of the structure and function of the nervous system. They are notoriously challenging to study due to their multifaceted nature and varied pathological features. Single-cell sequencing technologies have been developed and are powerful techniques for understanding the molecular basis of many pressing scientific questions such as the causality and development of Alzheimer's Disease (AD). These technologies, namely single-cell RNA sequencing (scRNA-seq) and single-cell Assay for Transpose-Accessible Chromatin sequencing (scATAC-seq), offer us an understanding of a cell’s state as a phase-space determined by chromatin accessibility and gene expression. Single cell data like this is extremely high dimensional; on the scale of 10s or 100s of thousands of cells, each with 10s of thousands of “features,” which represent genes or chromatin regions. Because of this, lower dimensional representations of these cells and clusters within them are valuable to help simplify our view of the data and extract signals. Moreover, in the context of cells characterized by biomarkers and stemming from patients with varying neurodegenerative diseases, it is in our interest to explore cell neighborhoods and embeddings to investigate if they properly represent the biological underpinnings of such disease. 

## Graph Neural Networks (GNNs) as an architecture and their application to single-cell analysis 
Graph Neural Networks (GNNs) are a class of deep learning models that are specifically designed to handle data that is structured as a graph, which extends the principles of neural networks to handle the concept of graph topology. In GNNs, each node (which in this application represents cells) aggregates information from graph neighbors through transformation and pooling steps, which results in a model whose representation captures node level and graph level features. Relevantly, GNNs generate lower dimensional embeddings of the input data, which provides a compact and informative representation of high dimensional data such as single-cell RNA data. 

The scGNN package specifically applies these principles of GNNs to single-cell genomics, treating cells as nodes in a graph and the edges as a measure of similarity in the transcriptome of two cells. scGNN performs two main functions: clustering and imputation. The architecture is as such:

- Feature Autoencoder: Generates low-dimensional representation of gene expression, which is the foundation for a cell graph.
- Graph Autoencoder: Learns a topological representation of the aforementioned cell graph,  which is the foundation for cell type clustering.
- Cluster Autoencoders: There is an autoencoder for each cell type that reconstructs gene expression values.
- Imputation Autoencoder: Recovers imputed gene expression values. 

## Intro to the Data

The [dataset](https://www.sciencedirect.com/science/article/pii/S009286742300973X?ref=pdf_download&fr=RR-2&rr=834b08acfbd66ac7) being presented is a scRNA-seq atlas of the aged human prefrontal cortex. It consists of 2.3 million cells sampled from 427 individuals over a varying range of Alzheimer’s pathology and cognitive impairment.  The subset of this data being analyzed in this project are the 19 samples that had multiome sequencing conducted, although only the scRNA-seq was used for this analysis (excluding the scATAC-seq). This was approximately 100 thousand cells and originally 36 thousand genes that are categorized into three diagnoses: no AD, early AD, and late AD based on biomarkers like amyloid plaque and niareagan score. 

## Applying scGNN to our AD scRNA-seq data


I began by processing the raw sequencing data into a csv format that would be suitable as input to the pipeline. I then ran preprocessing on this data, which consists of log transformation, filtering out low quality/sparse genes and cells, and subsetting to the top 2000 highly variable genes by variance. I then ran the actual imputation and clustering pipeline with the following parameters: EM-iteration=10, Regu-epochs=500, EM-epochs=200, cluster-epochs=200, quickmode=True, knn-distance=euclidean. The result of training is a imputed cell matrix, a cell graph, cell type clusters, and the actual embeddings of the cells themselves. These results provide the foundation for the next layer of analysis. 

## Visualizing the Degree Distribution of the Cell Graph

The figure below is a histogram that represents the number of other cells each cell in the dataset is connected to in the cell graph as computed by the Graph Autoencoder. We can see that the distribution is skewed right, which tells us that most cells are connected to a relatively few other cells, which could indicate a particularly heterogeneous cell population. However, there are a select few that have substantially higher number of connections, which could represent some sort of “hub” cells. 

![](/assets/img/2023-12-12-scRNA-GNNS/degree.jpeg)


## Understanding Cell Clusters in the Embedding Space
The next approach was a detailed analysis of the clusters generated by the graph architecture by comparing to clusters generated on the imputed output data. This is important in visualizing the efficacy of the GNNs embeddings in delineating cell types compared the clusters derived from traditional methods on the imputed data, which included all 2000 highly variable genes (HVGs). The steps are as following:

1. Computing Neighbors: Step 1 is to compute the neighbors for each cell, which as a reminder explains gene expression similarity between cells. 
2. Principal Component Analysis (PCA): The subsequent step is to compute PCA on the data, which is a dimensionality reduction technique. 
3. Louvain Clustering: After PCA, I used Louvain clustering, which is widely used in scRNA-seq analysis for clustering cell types, and tuned the resolution to match a similar number of clusters as generated in scGNN. 
4. UMAP Visualization: To visualize clusters, I used Uniform Manifold Approximation and Projection (UMAP), which is a dimensionality reduction technique that allows us to visualize the cell data in 2-dimensions, colored by cluster. I colored the UMAP first by the clusters generated on the embedded data by scGNN and then by the PCA/Louvain clusters. 

In the figures below, we see the result of computing cell type clusters based on data embedded by the feature and graph autoencoder versus using the traditional method of PCA then Louvain clustering. While they resulted in slightly different number of clusters, it is interesting to see that the traditional method appears to outperform the GNN in terms of separating clusters in the embedding space. Further analysis on the differentially expressed genes (DEGs) in each cluster would need to be done to confirm which cell type each cluster truly represents. Only then would we be able to determine the accuracy of each, but from a visual perspective in UMAP space, the GNN clusters are less consistent. 

![](/assets/img/2023-12-12-scRNA-GNNS/pca_louvainclusters.jpg)
![](/assets/img/2023-12-12-scRNA-GNNS/scGNNclusters.jpg)


## Exploring Alzheimer’s Related Gene Contributions to the Embedding Space

Deep learning techniques and architectures like VAEs and GNNs are promising and seemingly relevant techniques for topics like single-cell genomics where data is extremely high dimensional and sparse. However, these complex algorithms beg the question of whether and how they represent the underlying biology, especially in the context of diseases like Alzheimer’s. Fortunately, while still incurable, AD has been extensively researched, and is strongly associated with a number of hereditary genes, mutations, and misfolded protein aggregates. This known research provides a robust benchmark when applying new techniques to AD data. When trying to implicate new genes or represent genes (features) in a lower dimensional embedding space, it is usually a good sign to check whether the known biomarkers of AD are also being predicted or also being represented. In our case, these embeddings provide the opportunity to see if the model captures the relevant biological information, which can then provide some level of validation to any other genes that are also being represented. 

To explore this further, I performed correlational analysis between the gene expression matrix from the imputed data and the “expression” values derived from the embedding dataframe. By focusing on the top 1% (20 genes) of genes that had the highest correlation for each embedding, I identified any biologically relevant genes that were being represented in the embedding. Below is a list of the AD relevant genes that showed up as being highly represented in this embedding space. 


- APOE: This gene, particularly the e4 allele, is the most widely known genetic risk for late onset Alzheimer’s Disease. This allele is responsible for about half of all AD cases 
- APP: This gene is called Amyloid Precursor Protein. You might recognize amyloid, which is the main hallmark of AD when it misfolds and becomes aggregate plaque in the brain. Abnormal cleavage of APP leads to an increase in amyloid plaque accumulation. 
- SORL1: Genetic mutations of this gene are associated with AD because of its role in recycling APP. 
- BIN1: Bridging integrator 1 has been implicated in many AD GWAS studies and has been found to influence the spread of tau, which is another hallmark of AD when misfolded, leading to neurofibrillary tangles. 
- CLU: Clusterin has been implicated in AD for its role in clearing amyloid-beta plaque from the brain. 

For example, in the figures below you can see that APOE falls into the genes with the highest correlation for embedding number 24, with a correlation of 0.79, and APP falls into those for embedding number 5 with a correlation of 0.79 as well. 

![](/assets/img/2023-12-12-scRNA-GNNS/embedding5.jpg)
![](/assets/img/2023-12-12-scRNA-GNNS/embedding24.jpg)


## Wrapping it up

I hope this analysis has demonstrated the potential of combining advanced computational methods in deep learning with with foundational biological data like scRNA-seq on AD to unravel long standing questions we have in the field. 

## Future Analysis 
Due to computational time, I elected to train the model on the entire dataset. Future work could include training the model on subsets of the data separated by the different level of AD pathology, which would give a slightly more nuanced understanding of disease progression and how that is reflected in the embedding space of each diagnosis category. 
