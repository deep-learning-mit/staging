---
layout: distill
title: Comparing Clustering in Representation Space for Plankton Images 
description: Plankton imaging devices are becoming a key method for gathering in-situ data about plankton communities. These instruments can produce millions of images so automatic processes are necessary for extracting information from the images, making the application of machine learning to plankton data a key step in advancing the study of ocean biogeochemistry. In this project I will explore how the representation of plankton images in the latent space differs between classic supervised learning, the unsupervised contrastive learning method descriped in SimCLR, and a modified supervised contrastive learning method. 
date: 2023-11-11
htmlwidgets: true

# Anonymize when submitting
# authors:
#   - name: Anonymous

authors:
  - name: Barbara Duckworth
    url: "https://github.com/barbara42"
    affiliations:
      name: MIT


# must be the exact same name as your blogpost
bibliography: 2023-11-11-plankton-representation.bib  

# Add a table of contents to your post.
#   - make sure that TOC names match the actual section names
#     for hyperlinks within the post to work correctly.
toc:
  - name: Equations
  - name: Images and Figures
    subsections:
    - name: Interactive Figures
  - name: Citations
  - name: Footnotes
  - name: Code Blocks
  - name: Layouts
  - name: Other Typography?

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
# Introduction 

Plankton are fundamental to ocean ecosystems, serving as the primary producers in marine food webs and playing a critical role in our planet's carbon cycle <d-cite key="falkowski_role_1994"></d-cite>. Understanding their distribution and behavior is vital for assessing ocean health and predicting environmental changes <d-cite key="treguer_influence_2018"></d-cite>. 

Marine imaging devices, such as the Imaging Flow Cytobot, produce millions of datapoints during expeditions, making machine learning methods necessary for analyzing their output <d-cite key="olson_submersible_2007"></d-cite><d-cite key="sosik_automated_2007"></d-cite>. This project aims to advance this analysis by exploring representation learning for plankton images. 

Representation learning, where the system learns to automatically identify and capture the most relevant features of the data, can provide insight into morphological patterns that are not immediately obvious, and improve classification of the organisms<d-cite key="bengio_representation_2014"></d-cite>. 

Using a new dataset from a 2017 North Pacific research cruise<d-cite key="white_gradients2-mgl1704-ifcb-abundance_2020-04-01_v10_2020"></d-cite>, I will compare the effectiveness of classic supervised learning, unsupervised contrastive learning as described in SimCLR, and a modified supervised contrastive learning method. 

This exploration will potentially offer insights into the latent space representation of plankton imagery, enhancing our understanding of these crucial organisms and improving the efficiency of ecological data processing.

# Dataset  

{% include figure.html path="assets/img/2023-11-11-plankton-representation/fig-IFCB-examples.png" class="img-fluid" %}

Data for this project was gathered using the Imaging FlowCytobot (IFCB), an in-situ automated submersible imaging flow cytometer. It generates images of particles within aquatic samples between the size of 10 to 200 microns. These particles might be detritus, organisms, or beads used for calibration <d-cite key="olson_submersible_2007"></d-cite>. In this dataset, gathered during the Gradients research cruise in the North Pacific in 2017, there are 168,406 images. Each of the images has been classified by a taxonomic expert, Fernanda Freitas, as a part of the Angelique White's research group at the University of Hawaii<d-cite key="white_gradients2-mgl1704-ifcb-abundance_2020-04-01_v10_2020"></d-cite>. There are 170 unique classes, with a handful of classes dominating the data. The distribution of samples per class can be seen in the figure below. 

{% include figure.html path="assets/img/2023-11-11-plankton-representation/fig-dataset-class-distribution.png" class="img-fluid" %} 

# Methodologies 

## Baseline classifier 

To lighten the workload of the project, I will use a codebase developed by a team at WHOI that is designed to work with IFCB data - [WHOI IFCB classifier](https://github.com/WHOIGit/ifcb_classifier). This CNN classifier provides the option to use a number of namebrand models as the backbone, and I will be using ResNet <d-cite key="he_deep_2016"></d-cite>. The WHOI IFCB classifier first resizes all images, and I will be using the standard class balancing and augmentation techniques it provides.

## SimCLR

SimCLR is an unsupervised contrastive learning method for visual data. The labels of the dataset are ignored, and each individual image will have a "postive" pair generated from a set of augmentations. It is then assumed that all other images in a batch are "negative". Positive pairs are pulled together and pushed apart from the negatives in representation space using the NT-Xent (normalized temperature-scaled cross entropy) loss function. For the project, the base encoder will be ResNet, and subsequent projection head will be a 2-layer MLP, with all other settings based on the defualts descriped in the SimCLR paper<d-cite key="chen_simple_2020"></d-cite>.

{% include figure.html path="assets/img/2023-11-11-plankton-representation/fig-simCLR-diagram.png" class="img-fluid" %}

The NT-Xent loss function is defined as follows:

$$
\ell_{i,j} = -\log \left( \frac{\exp(\text{sim}(z_i, z_j)/T)}{\sum_{k=1}^{2N} \mathbb{1}_{[k \neq j]} \exp(\text{sim}(z_i, z_k)/T)} \right)
$$


where $z_i$ and $z_j$ are the positive pairs, $z_k$ is a negative pair, and $\tau$ is the temerature parameter<d-cite key="chen_simple_2020"></d-cite>.


## Supervised Constrastive Learning

Instead of just using data augmentation to create a positive pair $x_j$ for image $x_i$, the positive match for image $x_i$ will be chosen from the pool of images with the same class label as $x_i$. Similarly, the negative matches will not be the full set of data, but rather only from classes that $x_i$ does not belong to. 

The resulting subervised normalized temperature-scaled cross entropy loss function (SNT-Xent) is defined as follows:

$$
\ell_{i,j} = -\log \left( \frac{\exp(\text{sim}(z_i, z_j)/\tau)}{\sum_{k=1}^{2N} \mathbb{C}_{[i, k]} \exp(\text{sim}(z_i, z_k)/\tau)} \right)
$$

where $\mathbb{C}_{[i, k]}$ is a function that evaluates to 1 if the class $C_j$ of the image $z_j$ is not the same as the class $C_i$ of the image $x_i$. 

$$
\mathbb{C}_{[i, k]} = 
 
\begin{cases} 
1 & \text{if } C_i \neq C_j \\
0 & \text{if } C_i = C_j \\
\end{cases}
$$

I am choosing not to pull all images from the same class together do to computational constraints. Instead, $z_j$ is randomly chosen from the pool of images with the same class as $z_i$. 

## Representation Space Analysis 

Latent space representations will be extracted from each of the models at the layer preceeding the last fully connected layer. 

I will use t-SNE (t-Distributed Stochastic Neighbor Embedding) to visualize the feature vectors<d-cite key="vanderMaaten2008tsne"></d-cite>. 

I will then use K-means clustering and compare the emergent groups to the labels in the dataset.<d-cite key="kanungo_efficient_2002"></d-cite>

# Aknowledgments

ChatGPT was used to create visualizations, write some paragraphs which were then edited, and generate latex equations. 