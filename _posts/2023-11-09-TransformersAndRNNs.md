---
title: "Transformers and RNNs: How do transformers implement recurrence?"
author: "Cassandra Parent"
date: '2023-11-09'
bibliography: 2023-11-09-TransformersAndRNNs.bib
output: html_document
---

# Transformers and RNNs: How do transformers implement recurrence?

Since their invention, [transformers have quickly surpassed RNNs in popularity](https://arxiv.org/abs/2311.04823) due to their efficiency via parallel computing [4]. They do this without sacrificing, and ofte improving, model accuracy. Transformers are seemingly able to perform better than RNNs on memory based tasks without keeping track of that recurrence. This leads researchers to wonder -- why? In this project I'll analyze and compare the performance of transformer and RNN based models. 

## Prior Work
 This project is inspired by [Liu et al](https://arxiv.org/abs/2210.10749) which explored how transformers learn shortcuts to automata. They did this both by mathematical proof and also through experimentation on synthetic data sets. Their primary conclusion is that transformers are able to universally approximate these complex functions in few layers by building simple parallel circuits. This leads to improvement in computational efficiency and also performance improvements [1]. This project acts as an extension by looking at real-world datasets from different applications and seeing if the conclusions change in the real-world. 


## Project Set Up
I decided to use three different datasets to compare how transformers and RNNs performed differently or similarly inn different context. All datasets are sourced via Kaggle. These data sets will be [protein prediction based on amino acid sequence](https://www.kaggle.com/competitions/cafa-5-protein-function-prediction/data), [ECG abnormality prediction](https://www.kaggle.com/datasets/shayanfazeli/heartbeatl), and [stock price prediction](ttps://www.kaggle.com/code/faressayah/stock-market-analysis-prediction-using-lstm). I decided to use Kaggle because they have a lot of resources on how to preprocess the data and some examples of projects built from the dataset to help me understand if my performance metrics are appropriate. 

## Analysis
I will start my analysis by building basic transformer and RNN models. I will also expand the proof in PSET 3 that compares the speed of transformers and RNNs and formalize my conclusions. 

I will then run my models against the datasets in the project set up to evaluate performance: both in time and in accuracy. I will adapt the experiments in Liu et al to these application datasets and test if these conclusions hold up. This will include testing known shortcomings of transformers such as [length generalization](https://arxiv.org/abs/2207.04901) [3]. I plan on using Python's time methods for these experiements to measure the time RNNs versus transformers take to perform different tasks. 

The following questions will try to be answered: How many additional layers or recurrence are needed prior to RNNs becoming better? Are there tasks that RNNs do better on than transformers, why? What are the limitations in performance of transformers? Why can't a simpler model such as a MLP also keep track of performance since it's also a universal approximator (why is the transformer special)? 

I will compare the conclusions against the Liu et al paper [1]. 


## Additional Questions of Interest
These questions will be explored as time allows and may be prioritized differently based on the results of the initial analysis. 

Transfromers may do better in efficiency and accuracy in most machine learning applications, but those are not the only important metrics in the field. Which model is better at explainability or interpretability? Are there fairness differences between the models? 

These questions lie at the main finding of Liu et al where they find that typical transformers are able to find shallow shortcuts to learn automata [1]. Performance isn't lost here, but is something else lost? 

Here, I would aim to do both a literature search and a preliminary analysis to investigate these questions. I also find visualizations a particularly valuable learning tool, especially in blog posts so I would like to capture some sort of explainability information in a visual diagram. Rojat et al provides some [ideas for explainability](https://arxiv.org/abs/2104.00950) in time series DL techniques, and I would like to try to apply those in a way that can differentiate transformers and RNNs [2].  

## References 
1. Liu B, Ash JK, Goel S, Krishnamurthy A, and Zhang C. Transformers Learn Shortcuts to Automata. 2023, arXiv. 
2. Rojat T, Puget R, Filliat D, Ser JD, Gelin R, and Dias-Roriguez N. Explainable Artificial Intelligence (XAI) on TimeSeries Data: A Survey. 2023, arXiv. 
3. Anil C, Wu Y, Andressen A, Lewkowycz A, Misra V, Ramasesh V, Slone A, Gur-Ari G, Dryer E, and Behnam. Exploring Length Generalization in Large Language Models. 2022, arXiv. 
4. Qin Z, Yang S, and Zhong Y. Hierarchically Gated Recurrent Neural Network for Sequence Modeling. 2023, arXiv. 
