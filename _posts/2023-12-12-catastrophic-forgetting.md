---
layout: distill
title: A Method for Relieving Catastrophic Forgetting With Explainability
description: Using various explainability metrics to target, we freeze layers in CNNs to enable continual learning.
date: 2023-12-12
htmlwidgets: true

# Anonymize when submitting
# authors:
#   - name: Anonymous

authors:
 - name: Pieter Feenstra
   url:
   affiliations:
      name: MIT
 - name: Nicholas Dow
   url:
   affiliations:
      name: MIT

# must be the exact same name as your blogpost
bibliography: 2023-12-12-catastrophic-forgetting.bib  

# Add a table of contents to your post.
#   - make sure that TOC names match the actual section names
#     for hyperlinks within the post to work correctly.
toc:
  - name: Introduction
  - name: Related Works
    subsections:
    - name: Weight Changing Regularization
    - name: Architectural Changes
    - name: Explanability Metrics
  - name: Methods
    subsections:
    - name: Model Type
    - name: Saliency Mapping
    - name: Filter Visualization
    - name: Training Procedure and Dataset Selection
  - name: Results
  - name: Discussion
    subsections:
    - name: Takeaways
    - name: Limitations

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

With recent advancements in deep learning, the intelligence of computers is quickly rivaling that of humans. GPT-4, with significant size and data, is able to score in the 90th percentile of the BAR, 88th percentile of the LSAT, and the 92nd percentile on the SAT <d-cite key="openai2023gpt4"></d-cite>. In dermatology, sophisticated computer vision models have outperformed trained professionals in diagnosing skin diseases and cancer <d-cite key="jeong2023deep"></d-cite>. Despite this substantial computational advantage, neural networks notably lag behind humans in their capacity for continuous learning, a skill essential for any intelligent entity. Particularly, they suffer from catastrophic forgetting, a phenomenon in which the learning of a new objective significantly degrades performance on prior tasks.

The human brain is able to protect itself from conflicting information and reductions in performance on previous tasks using complex mechanisms involving synaptic plasticity <d-cite key="hadsell2020embracing"></d-cite>. In essence, the brain is able to self regulate the strength of its connections, allowing for neurons to become less activated according to their memory and relevance. This ability has been attributed for the unmatched ability to learn in humans, which has allowed for humans to show improvement in skill on nearly any motor task given training, while still remembering previous information <d-cite key="green2008exercising"></d-cite>. This, then, is highly desirable for neural networks. 

In contrast to the human’s ability to learn, neural networks significantly alter their parameters when learning a new task. In effect, the network's understanding of previous tasks is overwritten. This poses a great barrier to the creation of artificial general intelligences, which ultimately depend on continual, life-long learning <d-cite key="silver2011machine"></d-cite>.

With the rapid increase in size and complexity of models, the field of model explainability and the desire to understand exactly what models are doing has quickly grown. Specifically in the field of computer vision, effort has been made to understand how models make decisions, what information leads to this decision, and how they learn what to observe <d-cite key="haar2023analysis"></d-cite>. Methods such as saliency mapping, which displays the importance of aspects of an input image to predicting a class, filter visualization, which finds the most activating features for a given filter, and gradient class activation maps, which visualizes the gradients flowing into the final convolutional layer, have all significantly contributed towards the understanding of how models make decisions <d-cite key="adebayo2018sanity"></d-cite><d-cite key="erhan2009visualizing"></d-cite><d-cite key="selvaraju2017grad"></d-cite>. 

We propose to make use of these explainability methods for the intelligent freezing of filters of a convolutional neural network. Specifically, we use saliency maps and filter visualizations to consider what a model is observing to classify an image, and then decipher which filters are most strongly contributing to this. In this paper, we contribute the following: 1. We create a method for the ranking of importance of filters in a convolutional neural network. We expand and combine upon previous works in model explainability to understand which filters are most strongly contributing to positive predictions. 2. We create a method for the freezing of filters of a convolutional neural network according to these rankings. We do this by first training on one task, freezing filters according to importance, then retraining the same model on a novel task. In doing this, we both corroborate our ranking system and identify a new strategy for alleviating catastrophic forgetting. 


# Related Works
Continual learning and its core problem of catastrophic forgetting has gotten recent attention in deep learning research. It’s easy to understand why the goal of having a model that can adapt to new data without being completely re-trained is sought after, and there have been many approaches to the problem of aiding the model’s ‘memory’ of past tasks. Solutions range from attaching a significance attribute to certain weights in the model that regularizes change introduced by the new data to explicitly freezing weights via different metrics of the weights’ performance.
## Weight Changing Regularization
Elastic Weight Consolidation(EWC) approaches the problem of catastrophic forgetting by adding a ‘stiffness’ to the weights of previous tasks dependent on an approximation of the importance they had to previous task performance. The authors of ‘Overcoming catastrophic forgetting in neural networks’ <d-cite key="Kirkpatrick_2017"></d-cite>.  explain EWC as maximizing a posterior of the parameters over the entire dataset, and then splitting up the posterior into a loss over the new task and a posterior of the parameters over the old task. They model the posterior of the old data as a quadratic difference of the original parameters and the current ones multiplied by the Fisher information matrix, so minimizing this results in preventing parameters from changing too much from being predictable from the old task’s data. The authors of the original paper showed that EWC was effective at preventing CNN from forgetting how to classify the MNIST dataset and helping an RL model maintain performance in Atari games. However, EWC is an additional loss metric that must be calculated for each back-propogation and for each previous task; it’s also linear in the size of the output and therefore is prohibitive for high dimensional data.

Another technique that attempts to use a regularizing factor to slow the retraining of old task parameters is explicitly computing a importance metric for each neuron in the network<d-cite key="zenke2017continual"></d-cite>. The authors denote this method as “Synaptic Intelligence” as they drew their inspiration from the complex adaptation of synapses in the brain contrasted with the simple uni-scalar representation of neurons in a MLP network, and by allowing the network to account for the importance of they could help a neural network model the human behavior of continual learning. The metric they calculate as importance is based on 1) how much a parameter contributed to the reduction of loss over the entirety of training and 2) how much a parameter changed during training. They compared their performance to EWC and standard SGD on the MNIST dataset and found similar results to EWC while beating naive SGD as the number of consecutive tasks increased.
## Architectural Changes
A drastically different approach that a couple papers investigated was preventing interference between training runs by completely freezing the weights in parts of the model after completing a task’s training. The papers here differentiate themselves via the method they decide to freeze certain weights and layers. The earliest such paper we found was detailing a method called Packnet <d-cite key="mallya2018packnet"></d-cite>, where the weights they selected to keep via freezing was purely based on a certain percentage of the weights with the highest magnitude. They also made the decision to completely wipe the weights they did not freeze and then do a couple epochs of training on the model that was a mix of frozen and pruned weights. Their strategy achieved performance roughly equal to networks jointly trained on all the data at once and outperformed the naive strategy of simply retraining, validating a version of the freezing strategy.

Instead of simply measuring the magnitude of weights to decide what layers or specific weights to freeze, authors of a paper on catastrophic forgetting explainability paper use a custom metric to find a layer that scores highest on their metric and subsequently freeze all the layers prior to that layer <d-cite key="nguyen2022explaining"></d-cite> Their metric is an analysis of the difference in activation maps of a layer in the model pre- and post- training on the new task. They posit that this difference in activation is a measurement of how much a layer has forgotten how to activate in response to an input. Their reasoning for freezing the layers prior to the layer most changed by the new sample set is that the errors that induce catastrophic forgetting propagate throughout the network, so identifying the layer with the sharpest drop-off indicates that prior layers are to blame. This seemingly builds off an earlier paper  <d-cite key="nguyen2020dissecting"></d-cite> that uses a similar activation map difference scheme to delicate layers that change more easily during training and instead directly freezes those fragile layers rather than those prior. In both papers, their results for this technique are an improvement over their ‘fine-tuning’ baseline, but the more recent paper’s results were not that differentiated from just selecting a layer to freeze before training a new task.
## Explanability Metrics
There exists many other explainability metrics with which one can target layers prior to training on a new task to try to prevent interference, an interesting one being saliency maps. Saliency maps attempt to capture the importance of features of the input on the output of a deep neural network. In the domain of CNNs, this can be thought of both the pixels and larger features, such as a window on a car, that contribute to a correct classification; saliency maps are analogous to trying to map out what parts of an image a model uses to make  correct identification. A model of saliency maps we felt compelled enough to use in our project is that of  <d-cite key="srinivas2019fullgradient"></d-cite>, where their full-gradient approach creates saliency maps from the gradients of each layer. This strategy encapsulates the importance of both the inputs and the impact of neurons throughout the network on the saliency map. As parts of a neural network might suffer from varying degrees of catastrophic forgetting, being able to identify the saliency of individual neurons is a desirable quality in choosing a metric that explains catastrophic forgetting.

# Methods

## Model Type
We tested our method using VGG16. VGG16 is a deep convolutional neural network that has achieved impressive results on the ImageNet classification challenge, with a top-1 accuracy of 72% <d-cite key="simonyan2014very"></d-cite>. Its sequential nature lends itself well to explainability methods like saliency maps. Further, it is relatively quick to train, even given the constraints of Google Colab. All of these attributes were highly desirable, as it allowed for rapid iteration for hyperparameter tuning, computation of saliency maps and filter visualizations, and a direct way to compare the viability of our freezing method through image classification accuracy. To ensure that the model did not have inference on any tasks prior to training, we randomly initialized the parameters. 
{% include figure.html path="assets/img/2023-12-12-catastrophic-forgetting/vgg16.webp" class="img-fluid" %}
<div class="caption">
    Figure 1: Schematic of VGG16 Architecture
</div>

## Saliency Mapping
The computation of saliency maps is grounded in the principles of backpropagation. It follows a multi-staged procedure which uses gradients to consider the impact of each pixel in an image. First, it computes the partial derivatives of the target output with respect to individual segments of the input image. Then, it uses backpropagation to propagate error signals back to the input layer. It does this in order to identify the impact of pixels. It considers pixels with larger signals to have the greatest impact on the decision-making process. 
There are a bountiful number of papers which propose different improvements on the original saliency map. When selecting a procedure, we identified two key features necessary for a useful visualization. We believed that a saliency map must have a full explanation of why a model made its prediction. Secondly, we believed that rather than considering each individual pixel, it clusters pixels together to consider importance. After testing, we ultimately used full-gradient saliency maps <d-cite key="srinivas2019full"></d-cite>. Code for this method is publicly available on the GitHub created by the authors of this paper, fullgrad-saliency.

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/2023-12-12-catastrophic-forgetting/mug_raw.png" class="img-fluid" %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/2023-12-12-catastrophic-forgetting/mug_saliency.png" class="img-fluid" %}
    </div>
</div>
<div class="caption">
    Figure 2: Raw image and saliency map of a mug.
</div>

The essence of full-gradient saliency maps lines up directly with the key features that we identified. To begin, it defines importance in the input image as a change in the feature resulting in change in model output. It seeks to illustrate a full answer for the model’s output.  To this end, it considers both global and local importance of features in the input image, which results in a method which both weighs the importance of each pixel individually, but also considers the importance of different grouping of pixels. 

## Filter Visualization
In order to compute what different filters are looking at, we made use of the Convolutional Neural Network Visualizations GitHub repository, which is a useful library that has implementations of many popular explainability methods <d-cite key="uozbulak_pytorch_vis_2022"></d-cite>. Specifically, we used the implementation of a filter visualization method from the paper “Visualizing Higher-Layer Features of a Deep Network”, which uses backpropagation to maximize the activation of a given filter  <d-cite key="erhan2009visualizing"></d-cite>. With this, we can compute exactly what a filter is attempting to observe in an image. This method provides two different options for creating filter visualizations - one with gradient hooks, and one without. 

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/2023-12-12-catastrophic-forgetting/jar_feature_viz.png" class="img-fluid" %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/2023-12-12-catastrophic-forgetting/jar_raw.png" class="img-fluid" %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/2023-12-12-catastrophic-forgetting/jar_saliency.png" class="img-fluid" %}
    </div>
</div>
<div class="caption">
    Figure 3: Heatmaps of Feature Visualization(left), Actual Image(middle), Saliency Map(right)
</div>



## Training Procedure and Dataset Selection
We created two datasets from CIFAR-100 <d-cite key="erhan2009visualizing"></d-cite>. We randomly selected 20 classes out of the total 100 and then divided these groups into two. We filtered the images from CIFAR-100 so that only images of those classes were in our datasets. We did this to ensure that the tasks the model was attempting to learn were of equal difficulty. We chose CIFAR-100 because we believed it was of adequate difficulty for the VGG16 architecture. We normalized the data and augmented it with random horizontal flips and random croppings.
For the first instance of training, we trained using stochastic gradient descent for 10 epochs with a learning rate of 1E-3. We did not implement any regularization or early stopping, as it was not necessary given training losses and testing losses. After this training, we used the described methods for calculating saliency maps and filter visualizations. For each class in the first dataset, we calculated the most useful filters by comparing saliency maps for the class to all filters. We compared these through multiple metrics, including mean squared error and Pearson correlation. To account for the fact that different layers of convolutional neural networks capture different types of information, we froze some percent of filters in each individual layer rather than the entire model. We left this percent as a hyperparameter. 
To ensure fairness for each task, the second instance of training followed the same exact procedure as the first - the optimizer was stochastic gradient descent, we trained for 10 epochs, and used a learning rate of 1E-3. 


# Results 
For the sake of hyperparameter tuning and evaluating different strategies, we froze the datasets to be the first and second ten images of CIFAR-100. We sought to check how the number of filters we freeze changes performance across datasets, which metric is most useful in comparing saliency images to filter visualizations, and how viable this method is as compared to training on a single, larger dataset. Prior to the second round of training, the test accuracy on the first dataset was .4566 and the test accuracy on the second dataset was .1322. 
{% include figure.html path="assets/img/2023-12-12-catastrophic-forgetting/table1.png" class="img-fluid" %}
	
The impact of freezing varying numbers of filters is in line with expectation - the more filters you freeze, the less inference you can gain, but also the more you will remember your previous task. In the table above, we can observe that with 25% of the filters frozen, we perform the best on dataset 2, with an accuracy of 39.2%, but the worst on dataset 1, with an accuracy of 20.7%. In contrast, when 75% of the filters are frozen, we maintain an accuracy of 38.4%, but do not learn about the new task, with an accuracy of 25.7%. 
{% include figure.html path="assets/img/2023-12-12-catastrophic-forgetting/table2.png" class="img-fluid" %}

We found that mean squared error was the greatest metric for the comparison of saliency maps and filter visualizations, recording the highest average accuracy and also retaining much more information about the first dataset.  From the table, we can see that when freezing 50% of filters in the network and selecting using mean squared error, we do roughly ten percentage points worse on the first dataset, but gain nearly double this loss on the second dataset. When compared to the randomly frozen method, it performs significantly better on the first dataset. This suggests that the filters that we froze are actually more important for correct predictions than the average. It makes sense that Pearson correlation is not particularly useful for comparison - it is not able to take into account the spatial information that is crucial for this comparison.
{% include figure.html path="assets/img/2023-12-12-catastrophic-forgetting/table3.png" class="img-fluid" %}

Finally, we found that training tasks sequentially and using the freezing method with a comparison metric of mean squared error slightly outperforms training the model on a larger, combined dataset at once. With this method, the model performed five percentage points better on predicting classes in both the first and second dataset. It is important to note that the accuracy reported for the model trained on the combined dataset is just the average accuracy over all of the classes, not necessarily split by the datasets. Still, to ensure fairness, the training procedure used for the combined dataset was the same as for the sequential training procedure, but trained for twenty epochs at once rather than ten epochs at two different times. This result implies that intelligently freezing filters of a neural network can be a viable strategy for overcoming catastrophic forgetting, even if just in a smaller setting.


# Discussion
## Takeaways
Through using convolutional neural network explainability methods such as saliency maps and filter visualizations, we were able to observe key insights into the relevance of different filters in VGG16. Quantitatively, we were able to measure this by freezing these layers and observing how well performance persisted after training on a new task. We found that freezing filters according to the similarity of their visualizations to saliency maps retains significantly more inference on a previous task, suggesting that these filters were more relevant to the previous task. By freezing these weights, we were also able to outperform simply training on a larger dataset. 
We believe that more research should be directed towards applying explainability methods to achieve the objective of continual learning. Although there has been previous work in the past, these often rely on stopping catastrophic forgetting once it has been observed, rather than determining which parts of the network are too integral to a task to be retrained. 
## Limitations
Because we are completely freezing weights, it is unlikely that this method could be generalizable to an arbitrary number of tasks. Future works could explore the integration of elastic weight consolidation into our pipeline rather than stopping change entirely. Doing class by class freezing of filters also introduces a cap to the number of tasks that this method could generalize to and the number of classes that can be predicted in each task. During our research, we concluded that this approach was better than attempting to combine saliency maps, but future work could also explore how to effectively combine saliency maps to capture important aspects of each class. 
Further, this method relies on the comparability of saliency maps and filter visualizations. While it makes intuitive sense that a filter is more relevant if it is seeking the parts of an input that are most important for a correct prediction, it is not as simple as directly comparing the two. While we attempt to alleviate some of this issue by doing layer-by-layer freezing, future work could certainly explore better metrics for choosing filters, especially given the stark difference in performance when using something as simple as mean squared error compared to Pearson correlation. 
Finally, the computational overhead of the method in combination with the limitations of Google Colab resulted in an inability to train on high-resolution images and use larger models. We believe that using high-resolution images would significantly benefit the feasibility of the method, as saliency maps are much more clearly defined. We again leave this to future work, as we are unable to explore this path.


