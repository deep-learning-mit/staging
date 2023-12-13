---
layout: distill
title: Sample Blog Post
description: Your blog post's abstract.
  This is an example of a distill-style blog post and the main elements it supports.
date: 2022-12-01
htmlwidgets: true

authors:
 - name: Pieter Feenstra
   affliations:
    name: MIT
 - name: Nicholas Dow
   affliations:
    name: MIT

bibliography: 2022-12-01-distill-example.bib  

toc:
 - name: Motivation
 - name: Intuition
 - name: Implementation
 - name: Evaluation
 - name: References
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
Catastrophic forgetting, also known as catastrophic inference, is a phenomenon in machine learning which refers to the tendency of models to entirely lose inference on a previous task when trained on a new task. This can be attributed to the idea that the weights learned for the previous task are significantly altered during the learning process for the new task. In effect, the model’s understanding of the previous task is overwritten. If a perfect classifier, trained for classifying a dog versus a cat is trained to be a perfect classifier for classifying between a car and a truck, the model will lose valuable insights into the former task, even if updates in weights are not relevant to the new task. 

## Intuition
To test generalizability of networks and the retention of training of different tasks, we will attempt to target specific neurons in a trained network to “keep” or “drop-out” and then continue to train the modified model on new tasks.  By “dropping-out” we mean exclude them from training in the next tasks; the hope of this would be to choose a subset of neurons in the model to prevent further training on. After further training, we would check how much performance the model retained on the original task. We could extend this further to do the same “drop-out” across several tasks and then compare a model produced by “drop-out” to that of a model just trained on the whole dataset.
In terms of “drop-out” neuron choices, the most obvious choice would be the neurons most active in the classification task just trained with the idea that the most active neurons have the highest “contribution” to a correct classification. Another choice would be to choose neurons with the highest discriminative ability between the classes in the task, so the neurons that have the highest change in average value when classifying different samples.
	Within this general idea, there are a variety of avenues to explore: how many k neurons should be “dropped-out” or preserved from each training task? How does restricting the “drop-out” to only certain depths of the network affect performance? 

## Implementation
We will assess the proposed idea with an image classification task. We will make use of publicly available datasets from Kaggle, including datasets for the prediction of cats versus dogs, cars versus bikes, lions versus cheetahs, and children versus adults. For prediction, we will use a convolutional neural network with cross entropy loss. Convolutional neural networks are suited for image-related tasks and will allow for relatively easy computation of the most activated neurons, which we will consider to be filters with the highest magnitude output. After training a model, we will freeze gradient descent on the k filters, choosing the filters by different selection metrics, and train on new data. K will be a hyperparameter that will be adjusted to optimize performance. 
## Evaluation
We will evaluate this model through a comparative analysis with a baseline model. We will train both models on an initial dataset, freeze k parameters of our model, then retrain the models on a second dataset. We will then compare the accuracy on some test set of the initial data. We will repeat this with varying values of k. Ultimately, we will compare our model with a model trained on all data at once.
