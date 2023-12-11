---
layout: distill
title: Regularization Techniques for Attention Layers in Transformer Models
description: Attention layers are a integral part of the success of transformer models, but can also lead to overfitting on parts of input data when there is limited training data. Therefore, researchers have proposed methods to regularize attention layers to reduce overfitting and increase generalizability. This blog will analyze popular methods and explore novel approaches to regularization in attention layers.
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
  - name: Introduction
  - name: Background
  - name: Methodology
  - name: Results
  - name: Further Research
  - name: Conclusion

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

Transformer models are exeptionally popular and successful at completing many tasks. However, they can overfit to their training data if they are not given enough data to generalize. Frequently, part of the reason for overfitting is the overfitting of a self-attention layer, which highlights important tensors in the model. However, if there is not enough data, the attention layer can overfit to the training data and highlight some tensors too much. Therefore, researchers have proposed methods of regularizing attention layers. This regularization has many different approaches to solving this issue, from simply smoothing attention layers to encouraging multi-headed models to approach diffierent solutions. Therefore, there are differences in the effects of different regularization methods and some might perform better in different circumstances. There does not seem to be a standard approach to dealing with this form of regualrization and while many authors have claimed their regularizations have positive effects on training, there are few comparisions of methods. In this study, we will analyze previous work on self-attention and propose new regularization techniques to identify the advantages and disadvantages of differing models.

## Methodology

### Overall Architecture
We begin by implementing and benchmarking a vision transformer with no regularization. We had previously implemented a transformer model as part of 6.s898 problem set 3, so we used this as basis for our models. This transformer was easily modifiable and reletively simple and so it served as a good basis for a our adjustments. The framework of the architecture goes as follows.

1. Take a image and split it into patches of specified size. 
2. Embed these patches and add a positional encoding to their embedding.
3. Treat these embeddings as a sequence input to a transformer model.
4. Use a transformer model with multi-head self-attention to transform the input into some specified space. 
5. Use this output to classify the image.

For this specific model, we use a 6 layer transformer with 5 self-attention heads and a patch size of 4. We will be focusing on the multi-head self-attention phase of the transformer model. 


### Data

We use the CIFAR-10 and CIFAR-100 datasets for this research <d-cite key = "krizhevsky2009learning"></d-cite>. CIFAR-10 consists of 60,000 32x32 color images representing 10 different classes. These classes are airplanes, cars, birds, cats, deer, dogs, frogs, horses, ships, and trucks. They are evenly distributed, such that there are 6,000 images of each class. CIFAR-100 uses the same format, but instead has 100 evenly distributed classes. We split this data into training and test sets and tested the different forms of regularization. We found that our transformer model with no regularization could easily acheive near-zero error on the training data, but only acheived around 60% in test accuracy for the CIFAR-100 dataset and around 30% accruacy on the CIFAR-100 dataset. Therefore, the model is overfitting to the training data and testing regularization methods on this dataset could help the model generalize more on the test data.

### Regularization Methods

We tested the following regularization methods for our model. We tested 3 different categories of regularization, these being dropout based methods, weight smoothing methods, and cross-head methods. 

#### Dropout Based Methods

##### DropColumn <d-cite key = "zehui2019dropattention"></d-cite>:
This method uses dropout, a common regularization method used in fully connected neural networks, in self-attention layers. This can force the model to generalize better and not rely on specific inputs as much. The authors propose the following methodology to add dropout to the model, which is similar to standard dropout techniques in neural networks. To perform dropout, each column in the attention weight matrix is sampled from a bernoulli distribution with some probability. We set the sampled columns to zero weight during training. Therefore, we are able randomly drop columns in the attention weight matrix. 

##### DropHead <d-cite key = "zhou2020scheduled"></d-cite>:
We can also perform dropout on the heads within the multi-head attention layer. With this method, we completely drop heads during training to reduce reliance on particular heads and increase the generalizability of the model. This prevents the model from being dominated by a few attention heads.

#### Weight Smoothing Methods

##### Relaxed Attention <d-cite key ="lohrenz2023relaxed"></d-cite>:
This method smooths the attention weights in the self-attention layer to reduce overfitting. This helps reduce the magnitude of the highest attention scores. We do this by mixing in the uniform distribution to attention weights during training. We use some parameter $ \color{white} \gamma $ to evaluate different levels of mixing. Therefore, we apply the following function to our self-attention weights. 

$ \color{white} A[i,j] = (1-\gamma) \times A[i,j] + \gamma \times \frac{1}{T}, \quad \forall i, j \in [0,1,...,T]$

We use $ \color{white} \gamma = 0.1 $ for our experiments. This adds a low level of uniformity but prevents the model from only attending upon a small number of tensors during training. Therefore, this should limit the amount of overfitting that is possible.

##### Noise Injection

Noise injection has been used to regularize fully connected neural networks, but we have not found any literature that proposes using noise injection to regularize self-attention layers. We propose two methodologies to add regularization and robustness to our model training. We inject noise with the following formula.

$ \color{white} x_{noised} = x + \frac{1}{100} * median(x) * N(0,1) $

1. Overall Noise Injection:
The first methodology involves simply adding noise to the input during training. We do this by adding Guassian random noise to the input before calculating self-attention weights in each layer of the transformer.

2. Individual Head Noise Injection:
Our second proposed methodology takes advantage of the mutli-headed transformer design. We add different Gaussian random noise to each head, such that the heads will recieve different inputs. Therefore, the model must become more robust to different inputs.


##### Normalization

We propose adding the norm of the attention weight matrix to the loss function to the limit the emphasis of individual inputs. Therefore, this will smooth the weights and reward more uniform predictions. This should reduce overfitting and make the model more generalizable.

#### Cross-Head Methods

##### Decorrelation
We propose adding a decorrelation term to our loss function. The goal of this loss is the reward differences across attention heads. We begin by calculating the self-attention weights for all of the attention heads. We then compute the pairwise dot products of each head's attention weights. This will increase the loss if there are heads that are highly correlated. This will cause the heads of the network to differ from the other heads in the network and hopefully generalize better. Therefore, we use the following loss term.

$ \color{white} \text{Added Loss} = \sum_{i={0,...,H},j={i+1,...,H}} \frac{\text{sum}((\Lambda_i^T \Lambda_j)^2)}{\text{Number of elements in }\Lambda_i^T \Lambda_j}$, where H is the number of heads and $ \color{white} \Lambda_i$ is the ith attention head weights.

This method is inspired by another method, multi-head attention with disagreement regularization <d-cite key = "li2018multi"></d-cite>. However, the disagreement regularization method relies on calculating more differences than just the attention weight matrices, which is out the of scope of these experiments. 

## Results

### CIFAR-10

We begin by analyzing the training results on the Cifar-10 dataset. 
<div class="row mt-3">
    {% include figure.html path="assets/img/2023-11-06-attention-regularization/training_loss.png" class="img-fluid" %}
</div>
<div class="caption">
    Training Loss on the CIFAR-10 Dataset
</div>
<div class="row mt-3">
    {% include figure.html path="assets/img/2023-11-06-attention-regularization/training_accuracy.png" class="img-fluid" %}
</div>
<div class="caption">
    Training Accuracy on the CIFAR-10 Dataset
</div>
We see that most of the models, except for the dropout based models, acheive near zero error and perfect accuracy on the test set. Therefore, we see that the dropout term is stopping the model from perfectly memorizing the dataset. 

<div class="row mt-3">
    {% include figure.html path="assets/img/2023-11-06-attention-regularization/test_loss.png" class="img-fluid" %}
</div>
<div class="caption">
    Test Loss on the CIFAR-10 Dataset
</div>
Looking at the test results, the two dropout models have much lower loss achieved on the test dataset. The rest of the models have similar losses on the test dataset.
<div class="row mt-3">
    {% include figure.html path="assets/img/2023-11-06-attention-regularization/test_accuracy.png" class="img-fluid" %}
</div>
<div class="caption">
    Test Accuracy on the CIFAR-10 Dataset
</div>

 We see that the two dropout methods also have higher accuracy than the model without regularization. However, the decorrelation model has the highest test accuracy. Overall, the test dataset results are significantly lower than state of the art and a more advanced model may be needed to acheive better performance.
### CIFAR-100

We move on to training and testing the models on the CIFAR-100 dataset. This dataset has more classes and therefore fewer examples of each class. Therefore, the model finds it more difficult to generalize on the test dataset.
<div class="row mt-3">
    {% include figure.html path="assets/img/2023-11-06-attention-regularization/training_loss100.png" class="img-fluid" %}
</div>
<div class="caption">
    Training Loss on the CIFAR-100 Dataset
</div>
<div class="row mt-3">
    {% include figure.html path="assets/img/2023-11-06-attention-regularization/training_accuracy100.png" class="img-fluid" %}
</div>
<div class="caption">
    Training Accuracy on the CIFAR-100 Dataset
</div>
We see similar results to the CIFAR-10 dataset in training. The two dropout methods are unable to acheive perfect loss and accruacy but all other methods are able to. This includes the methods with added loss, that being the normalization method and the decorrelation method. This will depend on the parameters of the model and these models would have higher loss if we used more emphasis on the added loss.


<div class="row mt-3">
    {% include figure.html path="assets/img/2023-11-06-attention-regularization/test_loss100.png" class="img-fluid" %}
</div>
<div class="caption">
    Test Loss on the CIFAR-100 Dataset
</div>
We see that the two dropout methods have significantly lower loss on the test dataset, with all other methods performing similarly.

<div class="row mt-3">
    {% include figure.html path="assets/img/2023-11-06-attention-regularization/test_accuracy100.png" class="img-fluid" %}
</div>
<div class="caption">
    Test Accuracy on the CIFAR-100 Dataset
</div>
We again see consistent results with the CIFAR-10 dataset. The two dropout methods and decorrelation improve the accracy on the test set, while the others are about the same as without normalization. In this case, the drophead method performs the best.


## Further Reseearch

Further research is needed to further improve the generalizability of this transformer architecture. The model still has overfitting issues, even with high regularization and so more research with different architectures or regularization methods is needed to improve the study. Further comparison of regularization methods on alternative datasets and types of data, such as text, would also be valuable to look at.

## Conclusion

Regularization is an important tool to reduce overfitting and improve the generalizability of a model. The results show that adding various forms of regularization can improve the results of a model, but our implementations did not cause dramatic change to the ability of the model to generalize to the test set. Most of the models still had a very large gap between their training accruacy and test accuracy. However, we did see notable improvements for both the dropout models and the decorrelation model. The dropout models were the only models that added regularization such that the model could not perfectly memorize the training set. Therefore, their training accruacy was significantly lower but they also had higher test accuracy. Additionally, the decorrelation model was also successful. While the model followed a similar pattern during training to the model without regularization, the test accuracy was generally higher, suggesting the added error did force the model to learn different parameters. Therefore, based on these results, adding regularization can be helpful in improving the generalizability of transformer models, especially when they have limited data. The other methods, such as the noise based methods, normalization, and relaxation did not appear to have a significant effect on training or test outputs. It is likely that alternative parameters or architectures are needed to realize their effect. Lastly, while this analysis was only completed using vision transformers, different datasets or network architectures may have significantly different results. Therefore, these other regularization methods may be more successful in other contexts. However, these tests prove that there are circumstances in which regularization can have a beneficial effect on transformer performance and is therefore a worthwhile experiment when dealing with overfitting transformers.
