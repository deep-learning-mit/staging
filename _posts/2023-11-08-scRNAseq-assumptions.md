---
layout: distill
title: Examining assumptions in scRNA-seq foundation model pre-training (6.S898 Final Project)
description: Final project for MIT's Deep Learning (6.S898) class.
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
  - name: Introduction
  - name: Related work
  - name: Methods
  - name: Results
  - name: Discussion

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
If the fundamental building block of biology is the cell, then the fundamental building block of cells are genes.
Genes are small segments of DNA that encode the information to create a protein, and proteins are a diverse set of macromolecules that can perform a staggering range of chemical functions which, when taken all together, lead to the complex behavior of cells and the organisms they make up.
To create proteins from genes, an intermediate "data transfer" occurs through another molecule type known as RNA. This information flow of genes to RNA to proteins is typically referred to as "gene expression", and is so core to biology that it's also known as the "central dogma of molecular biology".

Due to the importance of gene expression, many technologies have been developed to make quantitative measurements of gene expression from cells.
One of the most prominent technologies is called single-cell RNA sequencing (scRNA-seq), which enables the measurement of the expression of all genes in a given cell, often measured across thousands of cells simultaneously <d-cite key="hwangSinglecellRNASequencing2018"></d-cite>.

{% include figure.html path="assets/img/2023-11-08-scRNAseq-assumptions/fig1_scRNA_seq_overview.png" class="img-fluid" %}
<div class="caption">
    Schematic overview of the scRNA-seq workflow. Figure sourced from <d-cite key="panMicrofluidicsFacilitatesDevelopment2022"></d-cite>.
</div>

Large scale scRNA-seq datasets have enabled the high-resolution profiling of different organs and tissues at the cellular level, uncovering diverse cell types, rare subpopulations, and dynamic gene expression patterns within complex tissues and organisms.
This technology has found applications in various fields, from developmental biology and immunology to cancer research and regenerative medicine.

While scRNA-seq has seen broad-scale adoption, many challenges remain.
In particular, an individual research experiment may focus on a particular cell or tissue type, and produce insufficient data to apply modern machine learning techniques. To supplement their data or to gain additional context, a researcher may wish to utilize data generated from other experiments or researchers. However, performing large-scale integration of datasets across samples, tissues, and experiments currently presents challenges of scalability and non-biological differences between datasets driven by experimental variability (colloquially referred to as "batch effects") <d-cite key="lahnemannElevenGrandChallenges2020"></d-cite>.

In parallel to the explosion of available scRNA-seq data, the machine learning field has seen an increasing trend towards "foundation models".
Foundation models are large-scale deep learning models pre-trained with vast amounts of data for the purposes of creating a generalizable representation of a particular datatype (e.g. text, images).
Given these developments, recent work has focused on developing scRNA-seq foundation models as an approach to solve the challenge of  integrating diverse sets of scRNA-seq datasets in a scalable and generalizable way <d-cite key="theodorisTransferLearningEnables2023"></d-cite> <d-cite key="yangScBERTLargescalePretrained2022"></d-cite> <d-cite key="cuiScGPTBuildingFoundation2023"></d-cite> <d-cite key="chenGeneptSimpleHardtoBeat2023"></d-cite> <d-cite key="yangGeneCompassDecipheringUniversal2023"></d-cite> <d-cite key="haoLargeScaleFoundation2023"></d-cite> <d-cite key="levineCell2SentenceTeachingLarge2023"></d-cite>. Beyond just integration, foundation models of gene expression hold great promise in contributing to a broader understanding of biology by learning a representation space of cellular state, which could also lead to a large impact in downstream applications such as *in silico* prediction of cellular responses to novel therapeutics.

In this post, we'll explore a fundamental assumption of three such models (Geneformer<d-cite key="theodorisTransferLearningEnables2023"></d-cite>, cell2sentence <d-cite key="levineCell2SentenceTeachingLarge2023"></d-cite>, and GenePT <d-cite key="chenGeneptSimpleHardtoBeat2023"></d-cite>), which is the assertion that a given gene expression profile can be well-approximated by a rank-value encoding of genes.

What exactly is a rank-value encoding? Well, a typical representation of gene expression is a vector $$ x \in \mathbb{R}^N $$, where $$ N $$ is the number of genes, and each entry is a measure of the corresponding gene's expression. In a rank-value encoding, gene expression is instead represented as a list of N strings, where the strings are gene names, and are ordered in descending order of the underlying gene expression value.

{% include figure.html path="assets/img/2023-11-08-scRNAseq-assumptions/fig2_rank_value_schematic.png" class="img-fluid" %}
<div class="caption">
    Standard encoding of gene expression values compared to a rank-value encoding.
</div>

The rank-value encoding provides an intuitive transformation of the continuous gene expression values into an English language sentence that is compatible with existing approaches for foundation models in the natural language processing (NLP) field. However, as can be seen above, the rank-value encoding also drops the information of the exact gene expression values. Hopefully by the end of this post, we'll have gained some intuition for how a rank-value encoding of gene expression could be hindering the development of foundation models for gene expression and see that this does play out in practice for a real scRNA-seq foundation model.


## Related work

### Overview of gene expression representations in foundation models
While we won't go into a full detailed comparison of different methods for constructing gene expression foundation models from scRNA-seq data, it's worth spending a little time discussing
the commonalities and differences of various approaches at a high-level.

The most important distinction for this post is between methods that use a rank-value encoding and those that don't. For methods that don't use a rank-value encoding, we see a further
distinction between methods that employ some form of value-binning, where continuous expression values are mapped to a discrete number of pre-specified bins, and those that don't. Methods that use a binning approach are scGPT<d-cite key="cuiScGPTBuildingFoundation2023"></d-cite> and scBERT<d-cite key="yangScBERTLargescalePretrained2022"></d-cite>. In both scGPT and scBERT, gene expression values are first binned to map the continuous values to a set vocabulary of tokens, and these tokens are then passed through an embedding layer to generate higher-dimensional representations.
In contrast, scFoundation<d-cite key="haoLargeScaleFoundation2023"></d-cite> calculates gene expression embeddings by first transforming continuous scalar values to a vector using a small MLP,
and then calculating a final embedding by using an attention mechanism over K learned vectors. While we won't cover the full details, schematics of the approaches can be seen below to get a sense of the overall architectures, and most importantly to see how they directly use the gene expression values as input.

{% include figure.html path="assets/img/2023-11-08-scRNAseq-assumptions/fig2_scGPT_schematic.png" class="img-fluid rounded z-depth-1" %}
{% include figure.html path="assets/img/2023-11-08-scRNAseq-assumptions/fig2_scBERT_schematic.png" class="img-fluid rounded z-depth-1" %}
{% include figure.html path="assets/img/2023-11-08-scRNAseq-assumptions/fig2_scFoundation_schematic.png" class="img-fluid rounded z-depth-1" %}
<div class="caption">
  Schematics of the various approaches that *do not* use a rank-value encoding (top to bottom): scGPT, scBERT, and scFoundation. Figures sourced from <d-cite key="cuiScGPTBuildingFoundation2023"></d-cite><d-cite key="yangScBERTLargescalePretrained2022"></d-cite><d-cite key="haoLargeScaleFoundation2023"></d-cite>.
</div>

On the other hand, we have the methods that we're most interested in for the purposes of this post: the ones that utilize a rank-value encoding of gene expression. These methods are: Geneformer<d-cite key="theodorisTransferLearningEnables2023"></d-cite>, GenePT<d-cite key="chenGeneptSimpleHardtoBeat2023"></d-cite>, and cell2sentence<d-cite key="levineCell2SentenceTeachingLarge2023"></d-cite>.
In Geneformer, gene expression values are first converted to a rank-value encoding and then used to train a Transformer-based model using a variant of a masked language modeling objective in which a set of genes at random ranks are masked, and the model must learn to predict the masked gene names.
In cell2sentence and GenePT, pre-trained auto-regressive language models (GPT-2 and GPT-3.5 respectively) are applied to the rank-value encoded list of genes to obtain cell-level embeddings that are then used for downstream tasks. Again, we won't dive into the full details of these approaches, but provide schematic overviews of them below.

{% include figure.html path="assets/img/2023-11-08-scRNAseq-assumptions/fig2_Geneformer_schematic.png" class="img-fluid rounded z-depth-1" %}
{% include figure.html path="assets/img/2023-11-08-scRNAseq-assumptions/fig2_genePT_schematic.png" class="img-fluid rounded z-depth-1" %}
{% include figure.html path="assets/img/2023-11-08-scRNAseq-assumptions/fig2_cell2sentence_schematic.png" class="img-fluid rounded z-depth-1" %}
<div class="caption">
  Schematics of the various approaches that *do* use a rank-value encoding (top to bottom): Geneformer, GenePT, and cell2sentence. Figures sourced from <d-cite key="theodorisTransferLearningEnables2023"></d-cite><d-cite key="chenGeneptSimpleHardtoBeat2023"></d-cite><d-cite key="levineCell2SentenceTeachingLarge2023"></d-cite>.
</div>

### Critical examinations of scRNA-seq foundation models
In light of the recent development of many approaches for scRNA-seq foundation models, researchers have also begun performing critical assessments of such models. One of the main value propositions of foundation models is generalization to new data in a few-shot or zero-shot manner. To test this hypothesis, Kedzierska et al.<d-cite key="kedzierskaAssessingLimitsZeroshot"></d-cite> benchmarked the performance of Geneformer and scGPT at two zero-shot tasks with novel datasets: cell clustering and integration of data across batches (i.e. batch effect removal) . They found that both methods underperformed compared to simpler baseline methods. Similarly, Boiarsky et al.<d-cite key="boiarskyDeepDiveSingleCell2023"></d-cite> compared scGPT and scBERT to logistic regressions in the context of cell type annotation, and also found that the simpler approach performed competitively.

However, both of the works discussed above focused on examining the performance of scRNA-seq foundation models as a black box, whereas to the best of our knowledge, there are no current works examining the fundamental assumptions implicit in these foundation model approaches. We hope to begin addressing that gap in this post. By understanding whether or not rank-value encoding well-approximates the real similarities and differences in gene expression across cell types, we hope to either validate this assumption or gain insight into future avenues for improving pretraining of such scRNA-seq foundation models.

## Methods

### Dataset
To perform our assessment of rank-value encoding, we'll work with the Tabula Sapiens dataset <d-cite key="consortiumTabulaSapiensMultipleorgan2022"></d-cite>. This scRNA-seq dataset is a reference-quality collection of nearly 500,000 cells from 24 organs, sourced from 15 normal human subjects. The Tabula Sapiens dataset provides a good testbed for our experiments, as the samples have been processed in a uniform manner, allowing us to ask how rank-value encoding performs in a "best case" scenario. In the future, it would be beneficial to see how rank-value encoding performs across datasets as well, as there may be advantages in terms of smoothing out experimental noise.

We use the final dataset from Tabula Sapiens, which has already been subjected to quality control assessment, filtering, and normalization. While we won't go into the details of their pipeline here, these are available in their manuscript for the interested reader. In line with typical scRNA-seq workflows, we also subset the full set of ~22,000 genes down to a subset of 2,435 genes that have been marked as "highly variable genes" (HVGs) in the Tabula Sapiens dataset. This is a fairly standard step in scRNA-seq data processing workflows, as many genes are constitutively expressed across cell types, and thus provide little information for distinguishing between cell types. Highly variable gene selection was performed by the Tabula Sapiens Consortium following the methods and recommendations in Seurat<d-cite key="stuartComprehensiveIntegrationSingleCell2019"></d-cite>, a commonly used scRNA-seq data processing package.

{% include figure.html path="assets/img/2023-11-08-scRNAseq-assumptions/fig3_cell_type_hist.png" class="img-fluid rounded z-depth-1" %}
<div class="caption">
  Number of cells per cell type. Note that the majority of cell types have ~1000 examples, but that there's a long tail of highly represented cell types with up to 35k examples.
</div>

Additionally, since the Tabula Sapiens dataset is quite large and also has some cell types that are disproportionately represented, as shown above, we'll also subset the data to get a more tractable dataset for experimentation. To do so, we'll focus on cell types with 500 or more examples, and then further randomly subsample to 500 cells per type. This leaves us with 89 cell types<d-footnote>acinar cell of salivary gland,
 adventitial cell,
 b cell,
 basal cell,
 basal cell of prostate epithelium,
 basophil,
 bladder urothelial cell,
 capillary aerocyte,
 capillary endothelial cell,
 cardiac endothelial cell,
 cardiac muscle cell,
 cd24 neutrophil,
 cd4-positive alpha-beta t cell,
 cd4-positive helper t cell,
 cd4-positive, alpha-beta memory t cell,
 cd4-positive, alpha-beta t cell,
 cd8-positive alpha-beta t cell,
 cd8-positive, alpha-beta cytokine secreting effector t cell,
 cd8-positive, alpha-beta cytotoxic t cell,
 cd8-positive, alpha-beta memory t cell,
 cd8-positive, alpha-beta t cell,
 classical monocyte,
 club cell,
 club cell of prostate epithelium,
 conjunctival epithelial cell,
 corneal epithelial cell,
 corneal keratocyte,
 dendritic cell,
 dn1 thymic pro-t cell,
 dn3 thymocyte,
 duct epithelial cell,
 endothelial cell,
 endothelial cell of artery,
 endothelial cell of lymphatic vessel,
 endothelial cell of vascular tree,
 enterocyte of epithelium of large intestine,
 enterocyte of epithelium of small intestine,
 epithelial cell,
 erythrocyte,
 erythroid progenitor,
 eye photoreceptor cell,
 fibroblast,
 fibroblast of breast,
 granulocyte,
 hematopoietic stem cell,
 hepatocyte,
 immature enterocyte,
 immune cell,
 innate lymphoid cell,
 intermediate monocyte,
 keratinocyte,
 kidney epithelial cell,
 luminal cell of prostate epithelium,
 luminal epithelial cell of mammary gland,
 lung ciliated cell,
 macrophage,
 mast cell,
 mature enterocyte,
 mature nk t cell,
 memory b cell,
 mesenchymal stem cell,
 monocyte,
 myeloid cell,
 myofibroblast cell,
 naive b cell,
 naive regulatory t cell,
 naive thymus-derived cd4-positive, alpha-beta t cell,
 naive thymus-derived cd8-positive, alpha-beta t cell,
 neutrophil,
 nk cell,
 nkt cell,
 non-classical monocyte,
 pancreatic acinar cell,
 pancreatic ductal cell,
 paneth cell of epithelium of large intestine,
 paneth cell of epithelium of small intestine,
 pericyte cell,
 plasma cell,
 regulatory t cell,
 respiratory goblet cell,
 skeletal muscle satellite stem cell,
 smooth muscle cell,
 stromal cell,
 t cell,
 thymocyte,
 type i nk t cell,
 type ii pneumocyte,
 vascular associated smooth muscle cell,
 vein endothelial cell</d-footnote> and 500 cells per type, for a total of 44,500 datapoints.

To interact with this data, we'll be using the `AnnData`<d-cite key="virshupAnndataAnnotatedData2021"></d-cite> and `scanpy`<d-cite key="virshupScverseProjectProvides2023"></d-cite> Python packages, which we won't cover in detail here but flag in case you're interested in working with such data in the future.

### Assessments
To assess how well a cellular state can be represented using a rank-value encoding of genes, we'll look at various measures of similarity in the raw gene expression space and the rank-value encoded space, and compare those measures both within cell types and between cell types. We'll calculate the following measures for all pairs of cells:
 1. Euclidean distance of UMAP-projected gene expression values
 2. [Spearman rank correlation coefficient](https://en.wikipedia.org/wiki/Spearman%27s_rank_correlation_coefficient)
 3. Euclidean distance of UMAP-projected Geneformer embeddings

 For each distance measure, we can then generate comparisons at the level of cell types by summarizing via the median of the pairwise distances, either within or between cell types. A schematic of this approach is shown below.

 {% include figure.html path="assets/img/2023-11-08-scRNAseq-assumptions/fig4_comparison_schematic.png" class="img-fluid rounded z-depth-1" %}
<div class="caption">
  Overview of method for computing distance measures between cells followed by summarization to generate comparisons at the level of cell types.
</div>

#### UMAP of gene expression values
The idea behind this comparison is to utilize the continuous gene expression vectors, but using UMAP (Uniform Manifold Approximation and Projection<d-cite key="mcinnesUMAPUniformManifold2020"></d-cite>) to approximate the kind of non-linear transformation one might learn using a deep neural network. To calculate these values, we perform UMAP embprojectionedding of the gene expression values using the `umap-learn` Python package with defaut settings and `n_components=5`. Once we have the per-cell projections, we calculate Euclidean distance between all pairs of cells.

#### Spearman rank correlation coefficients
The Spearman rank correlation is a non-parametric measure of correlation between two ranked lists, which we can leverage to obtain a direct comparison of rank-value encoded gene lists. To accomplish this, we first calculate a rank-encoding of each cell's gene expression, with identical values being assigned a [fractional rank equal to the mean of their ordinal ranks](https://en.wikipedia.org/wiki/Ranking#Fractional_ranking_(%221_2.5_2.5_4%22_ranking)). As the Spearman correlation is defined as the Pearson correlation on the rank-encoded lists, we can then directly calculate the Spearman correlations between all pairs of cells.

#### Euclidean distance of UMAP-projected Geneformer embeddings
To fully assess the effect of rank-value encoding in a deep learning model, we take this one step further by calculating the embeddings of our cells using Geneformer. We generate these embeddings by using their model and code as [hosted on HuggingFace](https://huggingface.co/ctheodoris/Geneformer) for tokenization and embedding of our gene expression vectors. For each cell $$i$$, we obtain an embedding vector $$ x_i \in \mathbb{R}^{256} $$. We further project these 256-dimensional vectors down to 5 dimensions using UMAP for consistency with the projections of the raw gene expression values described above, and then calculate Euclidean distance between all pairs of cells. The rationale here is that Euclidean distance between two points may be larger in a 256-dimensional space than a 5-dimensional space due the high dimensionality (i.e. "curse of dimensionality"). However, we do still see similar results when using the full 256-dimensional embedding vectors (see Appendix).

## Results

### Rank-value encodings preserve similarity between cell types
The first thing we can see from our results is that rank-value encodings do preserve similarity between cell types in a similar manner as distances generated from raw gene expression values. The figure below is generated by looking at the distributions of distances between pairs of cells from the same type ("within") or from different cell types ("between"). To provide a comparison at the level of cell types, we plot the median of each distribution rather than individual pairs of cells, i.e. the "within" group contains 89 data points and the "between" group contains $$ \frac{89 \times 88}{2} $$ data points.

{% include figure.html path="assets/img/2023-11-08-scRNAseq-assumptions/fig6_combined_measure_comparison.png" class="img-fluid rounded z-depth-1" %}
<div class="caption">
  Comparison of various similarity measures both within cell types and between cell types. Note that for the Euclidean distances (left and right), lower is more similar, whereas for rank correlation (middle), higher is more similar.
</div>

How should we interpret this? What we can observe is that all three measures maintain high similarity for cells from the same type and less similarity for cells from different types. Put another way, rank-value encodings do define a space in which different cell types tend to be distant and cells from the same type tend to be near each other. We can also say that this holds when using both a non-parametric measure of the rank-value encodings (Spearman rank-correlation) and also when using a deep learning model that operates on rank-value encoded gene vectors (Geneformer).

However, we do also see that the difference between the "within" and "between" cell type distances is more pronounced when using a non-linear function on the raw data compared to either of the methods operating on the rank-value encoded gene vectors. This difference will become even more clear as we look at joint distributions of our different measures in the next section.

### Raw gene expression values better preserve within cell type similarities

To gain further insight into how rank-value encodings compare to raw gene expression values, we can look at the joint distributions of our distance measures. Below we see
the joint distribution of our raw gene expression-based distances compared to the rank-correlation values, shown as a 2D histogram where each hex is colored according to
the number of points that fall within that bin.

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/2023-11-08-scRNAseq-assumptions/fig7_raw_umap_vs_rank_corr_within_type.png" class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/2023-11-08-scRNAseq-assumptions/fig8_raw_umap_vs_rank_corr_between_type.png" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Joint distributions of distances from UMAP of raw gene expression values compared to rank correlations, within cell types (left) and between cell types (right).
</div>

We can notice that within cell types, the rank correlation has a fairly wide dynamic range whereas the raw gene expression-based distance seems to show a
tighter packing. Between cell types, we can observe that the rank correlations largely clump up closer to zero but do mesh with the larger distances we see
with the raw gene expression-based measure.

Given that we see a spreading out of cells within a type using a rank correlation, the natural question becomes whether this holds when we use a deep learning
model that can learn a complex non-linear function of the rank encodings. That's exactly what we look at below where we perform a similar comparison, but swapping
out the rank correlation distance measure for the distance measure based on Geneformer embeddings.

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/2023-11-08-scRNAseq-assumptions/fig9_raw_umap_vs_geneformer_umap_within_type.png" class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/2023-11-08-scRNAseq-assumptions/fig10_raw_umap_vs_geneformer_umap_between_type.png" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Joint distributions of distances from UMAP of raw gene expression values compared to distances from UMAP of Geneformer embeddings, within cell types (left) and between cell types (right).
</div>

With the Geneformer embeddings derived from the rank-value encodings, we now see that the between cell type distances are better matched to the distances derived from raw
gene expression values. However, we still see that Geneformer embeddings are more spread out within cell types compared to the non-linear transform of the raw gene expression
values. To better understand why this might be the case, we propose one possible contributing factor in the next section.

### Sparsity of scRNA-seq data may drive loss of information in rank-value encodings
A key aspect of scRNA-seq data is its extremely high sparsity. When working with single cells, the amount of available RNA is already quite limited, and then each processing step, such as RNA isolation or sequencing, introduces technical noise and the possibility of "dropout events", where a gene's expression is not detected at all. Combined with the inherent stochasticity of gene expression, we're often left with data where the vast majority of genes have zero detected RNA molecules.

Shown below is a histogram of sparsity per cell in the full Tabula Sapiens dataset as well as in the subset of cells and genes we considered in the analyses above.

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/2023-11-08-scRNAseq-assumptions/fig4_sparsity_full_dataset.png" class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/2023-11-08-scRNAseq-assumptions/fig5_sparsity_subset.png" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Histogram of gene expression sparsity per cell for the full Tabula Sapiens dataset (left) and the subset of 44,500 cells and 2,450 genes we considered in previous analyses. Sparsity here is defined as the fraction of genes with zero observed RNA molecules.
</div>

While many methods for processing scRNA-seq data attempt to handle the high sparsity in a principled manner, most of the methods described here simply remove genes with zero observations from
consideration. In particular, scGPT, GenePT, and Geneformer all remove genes with zero observations from their inputs, and cell2sentence restricts itself to the 100 genes with the highest
expression per cell, effectively removing all genes with zero observations. While sparsity is at least partially driven by stochastic technical factors, there is undoubtedly
a biological contribution as well, which may be removed when dropping genes with zero observations. While this issue is not unique to rank-value encoding, we can see that all
of the methods we've discussed here that use rank-value encoding remove genes with zero observations, likely to circumvent the ambiguity in how one would enforce an ordering
on genes that all have zero observations.

## Discussion
To give a high-level summary, what we've seen in this post is that rank-value encodings are an appealing way to transform continuous gene expression
vectors into a format that's directly compatible with the foundation model architectures that have seen great success in natural language processing. However, they
also seem to lose some valuable biologlical information of cell types, particularly information concerning similarity of cells within a given type.

While we don't present a smoking gun for an exact characteristic of this loss of information, we present sparsity as a key challenge in scRNA-seq data, which may
be exacerbated when using rank-value encodings. We can also further hypothesize that rank-value encodings may be sensitive to small changes in gene expression values
from technical noise, which could cause a shifting of ranks and thus amplify the impact of said noise. Similarly, rank-value encodings lose the absolute quantification
of gene expression, and this loss of granularity may impact the model's ability to capture the cases where subtle differences in gene expression hold biological
significance.

From the perspective of downstream use cases, models based on rank-value encodings are also limited in their ability to explore the counterfactuals that may be
interesting in cases such as predicting cellular responses to a novel therapeutic. For example, if a drug were known to affect the expression of a single gene, but
not to the point where the ranking of this gene shifted, then such a model would be unable to explore the downstream effect of this drug on the expression of other
genes.

In terms of limitations, the work presented here is fairly superficial and is constrained both in terms of size of datasets and breadth of methods compared. To
perform a more robust comparison in the future, we would like to scale up this analysis to larger datasets, such as the full Tabula Sapiens dataset. We would also
like to more directly compare cell type similarities in the embedding spaces of other scRNA-seq foundation models, including those that do and do not utilize rank-value
encodings. A great follow-up would be to perform a head-to-head comparison of a model like scBERT to Geneformer on the full Tabula Sapiens dataset.

Additionally, we've also yet to explore the angle of robustness across datasets. It's possible that some of the shortcomings we've listed for rank-value encodings
may actually be benefits in the context of supppressing technical noise when integrating scRNA-seq datasets across studies, institutions, and experimental techniques.
Performing this comparison across datasets would be a valuable follow-up that would help paint a more full picture of the value of rank-value encodings in the context
of constructing foundation models for gene expression data.

While we've discussed many challenges in constructing foundation-scale models for gene expression data, it's worth closing this post with an optimistic reflection on
the potential value of such models. By training a deep learning model to construct a representation space of cellular state, we stand to create a powerful tool that will
help us gain a fundamental understanding of cellular biology and its underlying complex regulatory networks. Ultimately, such tools could help us unravel the genetics of
various diseases, paving the way for a new era of disease treatments and precision medicine.

## Appendix

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/2023-11-08-scRNAseq-assumptions/fig11_SUP_raw_umap_vs_geneformer_raw_within_type.png" class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/2023-11-08-scRNAseq-assumptions/fig12_SUP_raw_umap_vs_geneformer_raw_between_type.png" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Joint distributions of distances from UMAP of raw gene expression values compared to distances from raw Geneformer embeddings, within cell types (left) and between cell types (right).
</div>