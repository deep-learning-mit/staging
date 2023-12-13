---
layout: distill
title: Combining Modalities for Better Molecular Representation Learning
description:
date: 2022-12-01
htmlwidgets: true

# Anonymize when submitting
# authors:
#   - name: Anonymous

authors:
  - name: Andrei Tyrin
    affiliations:
      name: MIT

# must be the exact same name as your blogpost

bibliography: 2023-12-12-combining-modalities-for-better-representation-learning.bib  

# Add a table of contents to your post.
#   - make sure that TOC names match the actual section names
#     for hyperlinks within the post to work correctly.
# toc:
#   - name: Introduction
#   subsections:
#     - name: Different ways to represent molecules
#     - name: Architectures for different modalities

toc:
  - name: Introduction
    subsections:
    - name: Importance of molecular representation learning
    - name: Different ways to represent molecules
  - name: Methods
    subsections:
    - name: Data
    - name: Models
    - name: Training
    - name: Evaluation
  - name: Analysis
    subsections:
    - name: Comparison of different models
    - name: Nearest neighbors analysis
  - name: Conclusion
    subsections:
    - name: Results of modalities mixing
    - name: Future work

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

### Importance of molecular representation learning
Molecular representation learning (MLR) — on of the most important tasks in molecular machine learning, drug design, and cheminformatics. <d-cite key=mol_rep_review></d-cite> There are several essential problems in the field of molecular sciences that rely on the high quality representation learning such as molecular property prediction, <d-cite key=mol_prop_pred></d-cite> organic reaction outcomes, <d-cite key=reaction_pred></d-cite> retrosynthesis planning, <d-cite key=retrosynthesis></d-cite> and generative modeling. <d-cite key=generative_review></d-cite> Solving these problems is crucial for the development of new drugs, materials, and catalysts.

### Different ways to represent molecules
Learning molecular representations poses more complex challenges than in computer vision or natural language processing due to the diversity of approaches for the initial encoding of the molecular structure as well as assumptions that follows from the choice of the representation. There are primarily four ways to represent molecules: 
1. **Fingerprints**. One of the oldest ways to represent molecules in Quantitative structure–activity relationship (QSAR) modelling. Molecular fingerprints are binary vectors that encode the presence or absence of certain substructures in the molecule. Fingerprints were one of the first ways to get the initial representation of molecules in machine learning problems. <d-cite key=fingerprints_pred></d-cite>
2. **String representation** (e.g. SMILES strings). Another way to represent a molecule — encode the fragments of the molecule as tokens which form a string when combined. This initial representation is widely used in generative molecular modeling. <d-cite key=lang_complex_distr></d-cite>
3. **2-D graph**. Very popular and natural way to represent molecules — encode the atoms and bonds as nodes and edges of a graph. Given the developments in GNNs arhictecutres,<d-cite key=gnns_review></d-cite> this representation is widely used in molecular property prediction. <d-cite key=chemprop></d-cite>
4. **3-D graph**. The most informative way to encode the structure of the molecule — encode the atoms and bonds as nodes and edges of a graph, but also include the spatial information about the atoms and bonds. Obtaining the 3-D graph representation of molecules can be really challenging, but models that operate on 3-D graphs tend to achieve the best performance. There are different modeling approaches to 3-D graphs, including invariant and equivariant GNNs. <d-cite key="schnet,equiv_gnn"></d-cite>

Given this diversity of approaches, the goal of this work is to explore the different ways to represent molecules and how they can be combined to achieve better performance on the downstream tasks such as molecular property prediction. Another goal of this blog post is to analyze the learned representations of small molecules via comparison of nearest neighbors in the latent chemical space. For the latter problem we also analyze the representations learned by language models trained on SMILES strings.

## Methods

### Data
We use the QM9 dataset to train and evaluate our models. The dataset contains ~133k small organic molecules with up to 9 heavy atoms (C, N, O, F) and 19 different properties. QM9 is a popular benchmark dataset for molecular property prediction. <d-cite key=qm9></d-cite> In our work we focused on predicting the free energy $G$ at 298.15K. We split the dataset based on Murcko scaffolds <d-cite key=murcko></d-cite> to ensure that the same scaffolds are not present in both train and test sets. We use 80% of the data for training, 10% for validation, and 10% for testing. The target values are standardized to have zero mean and unit variance.

### Models
The illustration of the overall approach is presented in Figure 1.
{% include figure.html path="assets/img/2023-12-12-combining-modalities-for-better-representation-learning/approach.png" class="img-fluid" %}
<div class="caption">
    Figure 1. Illustration of the overall approach. We use different ways to represent molecules and train different models on these initial encodings.
</div>

We use the following models to learn the representations of molecules:
1. **Fingerprint-based model**. We use the Morgan fingerprints <d-cite key=morgan></d-cite> with radius 2 and 2048 bits. We learned multilayer perceptron (MLP) with 6 layers,layer normalization and varying number of hidden units (from 512 to 256).
2. **SMILES-based model**. We used Recurrent Neural Network (RNN) with LSTM cells, 3 layers and 256 hidden units to learn the representations of SMILES strings presented in QM9 dataset. The learned representations were utilized in the nearest neighbors analysis. Model is trained to predict the next token in the SMILES string given the previous tokens. The cross-entropy loss is used for training:
$$ \mathcal{L}_{\text{CE}} = -\sum_{t=1}^{T} \log p(x_t | x_{<t}) $$

3. **2-D graph-based model**. We used the Message Passing Neural Network with 4 layers, 256 hidden units, sum aggregation, mean pooling and residual connections between convolution layers to learn the representations of 2-D graphs of molecules that updates the nodes hidden representation according to the equation below:

$$
h_i^{\ell+1} = \phi \left( h_i^{\ell}, \frac{1}{|\mathcal{N}_i|}\sum_{j \in \mathcal{N}_i} \psi \left( h_i^{\ell}, h_j^{\ell}, e_{ij} \right) \right)
$$

4. **3-D graph-based model**. While there are many different architectures to model points in 3-D space, we decided to use one of the simplest architectures — E(n) Equivariant Graph Neural Network (EGNN) <d-cite key=egnn></d-cite> that is equivariant to rotations, translations, reflections, and permutations of the nodes. We used 4 layers, 256 hidden units, sum aggregation, mean pooling and residual connections between convolution layers to learn the representations of 3-D graphs of molecules that updates the nodes hidden representations according to the equations given in the Figure 1.

### Training
All models were trained with Adam optimizer with learning rate $1\cdot10^{-3}$, batch size 32, and 100 epochs. We additionally used `ReduceLROnPlateau` learning rate scheduler. We used the mean absolute error (MAE) as the metric for evaluation.

### Evaluation
We used several combination of modalities to evaluate the performance of the models:
1. MPNN + FPs: the model that uses the representation learned by the MPNN and MLP trained on fingeprints with 256 hidden units as the final layer. This model concatenates the representations learned by the MPNN and MLP and then uses MLP to predict the target value.
2. EGNN + FPs: similar to the previous model but uses the representation learned by the EGNN.
3. EGNN + MPNN: this model concatenates the representations learned by the EGNN and MPNN and then uses MLP to predict the target value.
4. MPNN + RNN: this model concatenates the representations learned by the MPNN and RNN (pretrained) and then uses MLP to predict the target value. RNNs encodings are static and are not updated during the training. This model was not able to converge during the training and therefore was not used in the final evaluation.
The results of evaluation of different models on the QM9 dataset are presented in Figure 2.
<div class="l-page">
  <iframe src="{{ 'assets/html/2023-12-12-combining-modalities-for-better-representation-learning/mae.html' | relative_url }}" frameborder='0' scrolling='no' height="600px" width="100%"></iframe>
</div>
<div class="caption">
    Figure 2. Different models' performance on the QM9 dataset. The models are trained on the same data, but with different representations. The number of parameters is displayed on top of each bar.
</div>

## Analysis
### Comparison of different models
From the bar plot results presented in Figure 2 it's clear that EGNN shows superior performance. Possible explanation to this is that the QM9 data labels were computed using computational techniques that actively use the 3-D structure of the molecules. Therefore, 3-D representation of molecules turns out to be the best for this task and EGNN is able to capture interactions that happen in 3-D space that are crucial for the prediction of the target value. Therefore, simple concatenation of hidden representations in some sense dilutes the information and leads to worse performance. Overall conclusion from this experiment is that combining modalities is not a trivial task and requires careful design of the whole model architecture. <d-cite key="modality_blending,molecule_sde"></d-cite>

### Nearest neighbors analysis
After the training of the models we performed the nearest neighbors analysis to compare the learned representations of molecules. We took the learned representations of the molecules in the test set and computed the nearest neighbors in the latent chemical space using cosine similarity. Additionally we plotted the PCA reduced representations (Figure 3) and analyzed the nearest neighbors for 4 different molecular scaffolds.
{% include figure.html path="assets/img/2023-12-12-combining-modalities-for-better-representation-learning/dl_pic3.png" class="img-fluid" %}
<div class="caption">
    Figure 3. PCA reduced representations of the molecules in the test set. The color of the points corresponds to the molecular scaffold.
</div>

There are several interesting observations from the nearest neighbors analysis:
1. In case of fingerprints reductions the nearest neighbors are far away from the queried molecules in the latent chemical space.
2. For the reduced learned representations of the molecules in the test set we can see that the nearest neighbors are very close to the queried molecules in the latent chemical space. This is expected as the models were trained to predict the target value and therefore the representations of the molecules that are close in the latent chemical space should have similar target values.
3. In the right bottom plot of figure 3, for the EGNN + FPs combination we can see very interesting pattern — the reduced chemical space reminds the combination of the reduced chemical spaces of the EGNN and FPs. EGNN's reduced chemical is more "sparse", while the representation that learned by MLP is more dense but much more spread out. Another interesting observation is that the combined chemical space is more structured due to the presence of some clustered fragments, which is not present in case of both EGNN and MLP.

Additionally we analyzed the nearest neighbors for 4 different molecular scaffolds. The results for 3 of them are present in Figure 4.
{% include figure.html path="assets/img/2023-12-12-combining-modalities-for-better-representation-learning/dl_pic4.png" class="img-fluid" %}
<div class="caption">
    Figure 4. Nearest neighbors for 3 different molecular scaffold instances. Top molecule for each cell is the closest molecule to the queried molecule in the latent chemical space, the bottom molecule is the second closest molecule.
</div>

From the Figure 4 we can make some additional observations:
- For the fingerprints similarity, molecules are very similar to the queried molecule. This is expected results because the molecules with the highest matches in the fingerprints are the most similar to the queried molecule. Although, for the third example the second closest molecule is not very similar to the queried molecule.
- MPNN, EGNN as well as their combination return the molecules that are very similar to the queried molecule. Because the model was trained to predict the target value, the nearest neighbors are molecules with similar target values (this is not guaranteed for the fingerprints similarity because substructures can be combined in different ways potentially leading to very different molecular properties).
- In case of MLP trained on fingerprints, the nearest neighbors can have very different scaffolds. This agrees with the performance of the model on the QM9 dataset — the model is not able to fully capture the molecular structure and therefore the nearest neighbors can have very different scaffolds even though the initial representations were the ones retrieving the most similar molecules (fingerprints).
- Interestingly, in case of RNN trained on SMILES strings, the nearest neighbors can have very different scaffolds. This result is expected because RNN was trained to predict next token in the sequence and therefore the nearest neighbors are the molecules with similar SMILES strings. For example, first molecule contains triple bond between two carbon atoms. In the case of the second closest neighbor for first scaffold instance there are two triple bonds between carbon and nitrogen atoms. The scaffold is different, but the SMILES strings are similar.

The overally observation is that the better the model performed during the supervised stage (except RNN) the more meaningful nearest neighbors it returns in a sense that the nearest neighbors are more similar to the queried molecule in terms of the molecular structure. Fingerprints similarity still returns very similar molecules and therefore can be used for the nearest neighbors analysis, but the results are not as meaningful as for the GNNs, which are able to capture the molecular structure more expressively.

## Conclusion
### Results of modalities mixing
Modalities mixing is a very interesting and promising approach for the problems in the field of molecular machine learning. However, architectures should be desinged carefully to achieve the best performance. In our work we showed that simple concatenation of the representations learned by different models can lead to worse performance on the downstream tasks.

### Future work
The obvious direction of future work — to experiment with different architectures for modalities mixing. Another interesting direction is to use the mixed modalities for the generative molecular modeling as string methods still perform better than majority of 3-D generative approaches even though the latter one is more natural. <d-cite key=benchmarking></d-cite> Therefore, it would be interesting to explore the combination of the string and 3-D graph representations for the generative modeling.