---
layout: distill
title: Dynamic Ensemble Learning for Mitigating Double Descent
description: Exploring when and why Double Descent occurs, and how to mitigate it through Ensemble Learning.
date: 2023-11-08
htmlwidgets: true

# Anonymize when submitting
# authors:
#   - name: Anonymous

authors:
  - name: Mohit Dighamber
    affiliations:
      name: MIT
  - name: Andrei Marginean
    affiliations:
      name: MIT

# must be the exact same name as your blogpost
bibliography: 2023-11-08-double_descent.bib  

# Add a table of contents to your post.
#   - make sure that TOC names match the actual section names
#     for hyperlinks within the post to work correctly.
toc:
  - name: Motivation
  - name: Related Work
  - name: Methods
    subsections:
    - name: Decision Trees
    - name: Random Forest
    - name: Logistic Regression
    - name: Support Vector Machines
    - name: Neural Networks
  - name: Evaluation
    subsections:
    - name: Software
    - name: Datasets
    - name: Computing Resources
    - name: Reproducibility Statement

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

## Motivation

There are many important considerations that machine learning scientists and engineers
must consider when developing a model. How long should I train a model for? What
features and data should I focus on? What exactly is an appropriate model size? This
last question is a particularly interesting one, as there is a bit of contention regarding the
correct answer between different schools of thought. A classical statistician may argue that,
at a certain point, larger models begin to hurt our ability to generalize, whereas a modern
machine learning scientist may contest that a bigger model is always better. In reality,
neither of these ideas are completely correct in practice, and empirical findings demonstrate
some combination of these philosophies.

This brings us to the concept known as **Double Descent**. Double Descent is the phenomenon
where, as a model’s size is increased, test loss increases after reaching a minimum, then
eventually decreases again, potentially to a new global minimum. This often happens in the
region where training loss becomes zero (or whatever the ’perfect’ loss score may be), which
can be interpreted as the model ’memorizing’ the training data given to it.
The question of ’how big should my model be?’ is key to the studies of machine learning
practitioners. While many over-parameterized models can miraculously achieve lower test
losses than the initial test loss minimum, it is fair to ask if the additional time, computing
resources, and electricity used make the additional performance worth it. To study this
question in a novel way, we propose incorporating **Ensemble Learning**.

Ensemble Learning is the practice of using several machine learning models in conjunction
to potentially achieve even greater accuracy on test datasets than any of the individual
models. Ensemble Learning is quite popular for classification tasks due to this reduced error
empirically found on many datasets. To our knowledge, there is not much literature on how
Double Descent is affected by Ensemble Learning versus how the phenomenon arises for any
individual model.

We are effectively studying two different types of model complexity: one that incorporates
higher levels parameterization for an individual model, and one that uses several models in
conjunction with each other. We aim to demonstrate how ensemble learning may affect the
onset of the double descent phenomenon. Possible results may include that the phenomenon
occurs at a smaller or larger level of model complexity, the increase in loss before the second descent is more or less steep, or that the behavior of the test loss curve changes in some other way.

These results can potentially be used by machine learning researchers and engineers to
build more effective models. If we find that an ensemble model mitigates the increase in test
loss or brings about a second descent sooner as we increase model size, that may be evidence
in favor of using ensemble methods for different machine learning tasks, assuming that the additional resources used to build and train an ensemble model do not supersede the costs
potentially saved by this method.

***

## Related Work

One of the first papers discussing double descent was Belkin et al. <d-cite key="belkin2019reconciling"></d-cite>. This paper challenged the traditional idea of the 'bias-variance tradeoff'. They showed that after the interpolation threshold (where the model fits perfectly to the training data), test error eventually began to decrease once again. 

Nakkiran et al. <d-cite key="nakkiran2021deep"></d-cite> expanded these findings to the realm of **deep** learning. In this work, double descent is shown to occur for both large models and large datasets. Additionally this paper demonstrates that, counterintuitively, adding more data at a certain point actually worsened the performance of sufficiently large models. This highlights the need for a new understanding for model selection for effectively generalizing to testing datasets. 

In his classic paper 'Bagging Predictors' <d-cite key="breiman1996bagging"></d-cite>, Breiman describes the concept of combining the decisions of multiple models to improve classification ability. This bootstrap aggregating, or 'bagging' technique, reduced variance and improved accuracy, outperforming the single predictors that comprised the ensemble model. 

Another paper that discusses ensemble learning is Freund et al. <d-cite key="freund1997decision"></d-cite>, which introduced the Adaptive Boosting (AdaBoost) algorithm. On a high level, this paper illustrates how boosting is especially effective when combining weak learners that are moderately inaccurate to create a strong learner. We intend to use this algorithm as the basis of our ensemble methods.

***

## Methods

For this project, we will be using the tool `make_classification` from sklearn.datasets to unearth the double descent phenomenon. At the moment, we intend to experiment with five models, as well as an ensemble of them: decision trees, random forest, logistic regression, support vector machines, and small neural networks. We choose these models because of their ability to be used for classification tasks, and more complicated models run the risk of exceeding Google Colab’s limitations, especially when we overparameterize these models to
invoke double descent.

We will describe methods of overfitting these five models below. However, based on
feedback from course staff, we may change the models used for our experiments as necessary.

### Decision Trees

To invoke double descent for decision trees, we can start with a small
maximum depth of our tree, and increase this parameter until the training loss becomes
perfect or near perfect.

### Random Forest

We can begin random forest with a small number of trees, and
increase this until we see the double descent phenomenon in our test loss.

### Logistic Regression

To intentionally overfit using logistic regression, we can gradually increase the degree of the features. We can start with polynomial 1 and gradually
increase this parameter.

### Support Vector Machines

We will experiment with increasing the ’C’ parameter
for SVM, which is inversely proportional to regularization of the model. By default, this is
set as 1 in scikit-learn, but by increasing this, we can create a closer fit to the training data.

### Neural Networks

We can start by initializing a neural network with a small number
of layers and a small number of nodes per layer. We can then increase either or both of these
two parameters to achieve perfect training loss, and hopefully a better test loss level.

***

## Evaluation

To evaluate the performance of ensemble learning for mitigating the loss increase and
expediting the second descent in overparameterized models, we can plot the loss difference
between the ensemble loss curve and each individual model’s loss curve, where we plot loss
over model size. We can report the statistical significance of this difference to judge the
efficacy of using ensemble learning.

### Software

To train and test these models, we will be using various machine learning
packages in Python, such as Scikit-learn, PyTorch and Tensorflow. Additionally, to read in
.csv datasets and clean them as necessary, we will be using data science packages such as
pandas. Additional imports commonly used for machine learning project such as numpy and
matplotlib will also be utilized.

### Datasets

We plan on using `make_classification` from sklearn.datasets for our project
to generate classification data. This tool is publicly available for experimentation and our
use of it does not pose any ethical or copyright concerns.

### Computing Resources

We will be implementing this project using CUDA and the
free version of Google Colab. If these computational resources prove to be limiting for the
original scope of our project, we can scale down the training time, model size, and dataset
size as necessary, with the permission and guidance of course staff.

### Reproducibility Statement

To ensure reproducibility, we will save the seed that `make_classification` utilizes so that our results can be verified with the exact dataset we used. Additionally, we will provide our code in our final writeup.