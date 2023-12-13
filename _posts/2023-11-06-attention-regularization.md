---
layout: distill
title: Regularization Techniques for Attention Layers in Transformer Models
description: Attention layers are a integral part of the success of transformer models, but can also lead to overfitting on parts of input data when there is limited training data. Therefore, researchers have proposed methods to regularize attention layers to reduce overfitting and increase generalizability. This blog will analyze popular methods and explore potential novel approaches to regularization in attention layers.
date: 2023-11-06
htmlwidgets: true

# Anonymize when submitting
# authors:
#   - name: Anonymous

authors:
  - name: Anonymous

# must be the exact same name as your blogpost
bibliography: 2023-11-06-attention-regularization.bib  

# Add a table of contents to your post.
#   - make sure that TOC names match the actual section names
#     for hyperlinks within the post to work correctly.
toc:
  - name: Proposal
    subsections: 
    - name: Methods
    - name: Data
    - name: Implementation

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

Transformer models are exeptionally popular and successful at completing many tasks. However, they can overfit to their training data if they are not given enough data to generalize. Frequently, part of the reason for overfitting is the overfitting of a self-attention layer, which highlights important tensors in the model. However, if there is not enough data, the attention layer can overfit to the training data and highlight some tensors too much. Therefore, researchers have proposed methods of regularizing attention layers. This regularization has many different approaches to solving this issue, from simply smoothing attention layers to encouraging multi-headed models to approach diffierent solutions. Therefore, there are differences in the effects of different regularization methods and some might perform better in different circumstances. There does not seem to be a standard approach to dealing with this form of regualrization and while many authors have claimed their regularizations have positive effects on training, there are few comparisions of methods. Therefore, I propose a study of these regularization techniques to identify the advantages and disadvantages of differing models.

### Methods
The following are various regularization methods that would be interesting to test.
#### Relaxed Attention <d-cite key ="lohrenz2023relaxed"></d-cite>:
This method smooths the attention weights in the self-attention layer to reduce overfitting. This helps reduce the magnitude of the highest attention scores.

#### DropAttention <d-cite key = "zehui2019dropattention"></d-cite>:
This method uses dropout, a common regularization method used in fully connect neural networks, in self-attention layers. This encourages the model to use more of the input, rather than just a few tensors.

#### DropDim <d-cite key = "zhang2022dropdim"></d-cite>:
This method is an adapted form of dropout, which drops part of the embedding dimensions. This forces the transformer to learn with some of its embedding dimensions erased. We can tune the number of dimensions that are dropped.

#### Multi-head attention with disagreement regularization <d-cite key = "li2018multi"></d-cite>:
Regularization can also be applied to mulit-head attention. Specifically, this method uses disagreement regularization to encourage each head to be different from each other head. The methodology uses different combinations of regularization on different parts of multi-headed attention.

#### Potential New or Other Regularization Techniques:
I will explore other potential attention regularization techniques and look into novel approaches for regularization.

### Data

I will use a variety of data to sufficiently compare the above methods. We have already implemented a transformer model in the problem sets and tested that model on the CIFAR-10 dataset, so I will experiment with CIFAR-10, as well as other image datasets. Therefore, I will look into using CIFAR-100 and MNIST. I would also like to experiment with text input, depending on project scope and timing. 

### Implementation

I will complete more research regarding different types of regularization and the code already available to use for testing. I will either implement these methods into a PyTorch transformer or use the transformer we implemented in the problem set, depending on the ease at which I can add attention regularization to PyTorch. Therefore, more experimentation is needed to determine exact implementations for the project.


