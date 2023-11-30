---
layout: distill
title: Examining assumptions in scRNA-seq foundation model pre-training (6.S898 Project Proposal)
description: Initial proposal for a final project for MIT's Deep Learning (6.S898) class.
date: 2023-11-08
htmlwidgets: true

# Anonymize when submitting
# authors:
#   - name: Anonymous

authors:
  - name: Robert Calef
    url: "https://en.wikipedia.org/wiki/Robert_Calef"
    affiliations:
      name: MIT

# must be the exact same name as your blogpost
bibliography: 2023-11-08-scRNAseq-assumptions.bib

# Add a table of contents to your post.
#   - make sure that TOC names match the actual section names
#     for hyperlinks within the post to work correctly.
toc:
  - name: Background
  - name: Proposed Work

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
If the fundamental building block of biology is the cell, then the fundamental building block of cells are genes.
Genes are small segments of DNA that encode the information to create a protein, where proteins are a diverse set of macromolecules that can perform a diverse range of chemical functions which, when taken all together, lead to the complex behavior of cells and the organisms they make up.
The information flow of genes to RNA to proteins is typically referred to as "gene expression", and is so core to biology that it's also known as the "central dogma of molecular biology".

Due to the importance of gene expression, many technologies have been developed to make quantitative measurements of gene expression from cells.
One of the most prominent technologies is called single-cell RNA sequencing (scRNA-seq), which enables the measurement of the expression of all genes in a given cell, often measured across thousands of cells simultaneously <d-cite key="hwangSinglecellRNASequencing2018"></d-cite>.
Large scale scRNA-seq datasets have enabled the high-resolution profiling of individual cells, uncovering diverse cell types, rare subpopulations, and dynamic gene expression patterns within complex tissues and organisms.
This technology has found applications in various fields, from developmental biology and immunology to cancer research and regenerative medicine.

While scRNA-seq has seen broad-scale adoption, many challenges remain.
In particular, an individual research experiment may focus on a particular cell or tissue type, and produce insufficient data to apply modern machine learning techniques. To supplement their data data or to gain additional context, a researcher may wish to utilize data from other experiments, but currently performing large-scale integration of datasets across samples, tissues, and experiments presents challenges of scalability and lack of generalization due to batch effects <d-cite key="lahnemannElevenGrandChallenges2020"></d-cite>.

In parallel to the explosion of available scRNA-seq data, the machine learning field has seen an increasing trend towards "foundation models".
Foundation models are large-scale deep learning models pre-trained with vast amounts of data for the purposes of creating a generalizable representation of a particular datatype (e.g. text, images).
Given these developments, recent work has focused on developing scRNA-seq foundation models as an approach to solve the challenge of  integrating a diverse set of scRNA-seq datasets in a scalable and generalizable way <d-cite key="theodorisTransferLearningEnables2023"></d-cite> <d-cite key="yangScBERTLargescalePretrained2022"></d-cite> <d-cite key="cuiScGPTBuildingFoundation2023"></d-cite> <d-cite key="chenGeneptSimpleHardtoBeat2023"></d-cite> <d-cite key="yangGeneCompassDecipheringUniversal2023"></d-cite> <d-cite key="haoLargeScaleFoundation2023"></d-cite>.

In this proposal, we aim to explore a fundamental assumption of three such models (Geneformer<d-cite key="theodorisTransferLearningEnables2023"></d-cite>, scGPT <d-cite key="cuiScGPTBuildingFoundation2023"></d-cite>, and genePT <d-cite key="chenGeneptSimpleHardtoBeat2023"></d-cite>), which is the assertion that a given gene expression profile can be well-approximated by a rank-value encoding of genes.
All three of these models use a pretraining objective in which raw scRNA-seq data is first preprocessed to achieve gene expression values and then genes are ranked in descending order of their expression values.
These rank-encoded lists of genes are then used for a variant of a masked language modeling objective, in which a set of genes at certain ranks are masked, and the model must learn to predict the masked gene names.
By understanding whether or not this rank-value encoding well-approximates the real similarities and differences in gene expression across cell types, we hope to either validate this assumption or gain insight into future avenues for improving pretraining of such scRNA-seq foundation models.

## Proposed work
To assess how well a cellular state can be encoded using a rank-value encoding of genes, we will proceed in two steps.
First, we will restrict our analysis to a single dataset: a recently released atlas containing scRNA-seq data from aged human prefrontal cortex, covering 2.3 million cells from 427 individuals, and representing a range of cell types<d-cite key="mathysSinglecellAtlasReveals2023"></d-cite>.
This dataset has been generated using a uniform protocol followed by an identical computational processing pipeline, thus reducing the likelihood of batch effects and allowing us to focus on the question of whether rank-value encoding accurately encodes cell type.
We will then proceed by generating rank-value encodings of genes for each sample in the dataset, and calculating pairwise rank correlation coefficients for the ranked gene lists between all pairs of cells.
Given the large size of this dataset, this may be computationally prohibitive, so we could also perform subsampling of the dataset, stratified by annotated cell type to prevent dropout of rarer cell types.
Given the pairwise rank correlation coefficients, we can begin asking question like: using a given rank correlation coefficient cutoff to call related samples, what fraction of a given cell's relations are of the same cell type? Of those that are not from the same cell type, are they from a biologically similar cell type?

While this initial analysis may already be revealing, we also want to consider the effect of rank-value gene encoding *across* datasets.
Given that a key value proposition of scRNA-seq foundation models is integrating diverse datasets in a generalizable way (i.e. without inadvertantly capturing batch effects), we would also like to see if the rank-value gene encoding provides any value in terms of mitigating spurious differences within a cell type across datasets.
To accomplish this, we can utilize a dataset that was previously released with the explicit purpose of benchmarking methods for handling batch effects in large-scale scRNA-seq dataset integration efforts <d-cite key="lueckenBenchmarkingAtlaslevelData2022"></d-cite>. Utilizing this dataset, we can again calculate pairwise rank correlation coefficients and ask what fraction of a given cell's relations are from the same cell type, biologically similar cell types, or completely different cell types. To more directly compare to an alternative of using raw gene expression values, we could also compare nearest neighbors in terms of rank-correlation coefficient to a set of nearest neighbors in raw gene expression space, and ask if either set displays a larger proportion of batch effect-driven neighbors.

We may find that the rank-value encoding does not well approximate cell type or that there are interesting corner cases that are not well captured. In this case, an interesting follow-up would be to modify the approach taken by Chen and Zou in genePT <d-cite key="chenGeneptSimpleHardtoBeat2023"></d-cite>, in which cell embeddings are calculated by directly inputing a rank-value encoded lists of gene names into an OpenAI text embedding model. Since such an approach doesn't rely on training or fine-tuning a new model, we could quickly iterate on modifications of their approach based on our findings to experiment with approaches to improve performance in simple downstream tasks like association between embeddings and underlying cell states.

