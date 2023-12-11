---
layout: distill
title: "LSTM vs Transformers for Time Series Modeling"
description: A comparison study between LSTM and Transformer models in the context of time-series forecasting. 
date: 2023-12-12
htmlwidgets: true

# Anonymize when submitting
# authors:
#   - name: Anonymous

authors:
  - name: Miranda Cai
    url:
    affiliations:
      name: MIT
  - name: Roderick Huang
    url:
    affiliations:
      name: MIT

# must be the exact same name as your blogpost
bibliography: 2023-12-12-time-series-lstm-transformer.bib

# Add a table of contents to your post.
#   - make sure that TOC names match the actual section names
#     for hyperlinks within the post to work correctly.
toc:
  - name: "Introduction"
  - name: "Related Work"
    subsections:
    - name: Effect of Dataset Size
    - name: Effect of Noisy Datasets
    - name: Effect of Multi-step Prediction
  - name: "Methodology"
  - name: "Experimental Results and Discussion"
    subsections:
    - name: Size of a Dataset
    - name: Amount of Noise in a Dataset
    - name: Output Size
  - name: "Conclusion"


---

# 6.S898 Final Project - LSTM vs Transformers for Time Series Modeling

By Miranda Cai and Roderick Huang

## 1. Introduction
In the context of time series forecasting, comparing Long Short-Term Memory (LSTM) networks to Transformers is a fascinating exploration into the evolution of deep learning architectures. Despite having distinct strengths and approaches, both LSTM and transformer models have revolutionized natural language processing (NLP) and sequential data tasks.


LSTMs, with their recurrent structure, were pioneers in capturing long-range dependencies in sequential data. While the accuracy of such models have been shown to be quite effective in many applications, training LSTM models takes a relatively long time because of the fact that they must remember all past observances. One faster alternative to LSTM models are transformers. Transformers are able to remember only the important bits of inputs using an attention-mechanism, and is also parallelizable making it much faster to train than recursive LSTMs that must be processed sequentially. 

With its recent development, people have started opting to use transformer based models to solve sequence problems that once relied on LSTMs. One significant example is for NLP use cases, where transformers can process sentences as a whole rather than by individual words like LSTMs do. However, since transformers have been around for less than a decade, there are still many potential applications that are yet to be deeply explored. Thus, we will explore the effectiveness of transformers specifically for time series forecasting which finds applications across a wide spectrum of industries including finance, supply chain management, energy, etc. 

Our goal is to realize which particular features of time series datasets could lead transformer-based models to outperform LSTM models. 

## 2. Related Work

With the growth of ChatGPT in the recent years, extensive research has been done across various NLP tasks such as language modeling, machine translation, sentiment analysis, and summarization, each aiming to provide comprehensive insights into when each architecture excels and where their limitations lie. While research on time series data exists, it hasn't garnered as much attention, so we aim to broaden this area of study.

### 2.1 Effect of Dataset Size
The size of a dataset plays an important role in the performance of an LSTM model versus a transformer model. A study <d-cite key="comparison"></d-cite> done in the NLP field compared a pre-trained BERT model with a bidirectional LSTM on different language dataset sizes. They experimentally showed that the LSTM accuracy was higher by 16.21% relative difference with 25% of the dataset versus 2.25% relative difference with 80% of the dataset. This makes sense since BERT is a robust transformer architecture that performs better with more data. As shown in the figure below from <d-cite key="comparison"></d-cite>, while LSTM outperformed BERT, the accuracy difference gets smaller as the perctange of training data used for training increases.
<p align="center">
  <img src="./assets/img/2023-12-12-time-series-lstm-transformer/dataset_size_research_fig.png">
</p>
While we perform a similar methodology which is discussed further in section 4.1, the major difference is in the type of data we test. Instead of measuring classification accuracy for NLP tasks, this study measures the mean squared error (MSE) loss for regression time series data. 

### 2.2 Effect of Noisy Datasets

Theoretically, LSTMs are more robust to noisy data due to its ability to capture local dependencies. On the other hand, the self-attention mechanisms in transformers propagate errors and may struggle with sequences that have a high degree of noise. Electronic traders have been recently attempting to apply transformer models in financial time series prediction to beat LSTMs <d-cite key="trading"></d-cite>. Largely focused on type of assets, the research showed that transformer models have limited advantage in absolute price sequence prediction. In other scenarios like price difference and price movement, LSTMs had better performance.

Financial data sets are known to be extremely noisy, and in addition, very hard to find due to their confidential nature. The application of <d-cite key="trading"></d-cite> gave inspiration to study how the "amount" of noisiness would affect the LSTM and transformer models. Discussed further in section 4.2, this study added various amounts of noise to a clean dataset to see how this would affect each architecture.

### 2.3 Effect of Multi-step Prediction
gonna relate to this paper: https://arno.uvt.nl/show.cgi?fid=160767
<d-cite key="multistep"></d-cite>
they train by filtering out data, less precision
so ours is novel by seeing if we could keep level of precision but predict into the future

## 3. Methodology

The dataset we will be using throughout this study is the Hourly Energy Consumption dataset that documents hourly energy consumption data in megawatts (MW) from the Eastern Interconnection grid system <d-cite key="dataset"></d-cite>. 

<p align="center">
  <img src="./assets/img/2023-12-12-time-series-lstm-transformer/energy_dataset_split.png" width="700">
</p>

We can utilize this dataset to predict energy consumption over the following features of a dataset.
- **Size of a dataset**: As discussed in Section 2.1 <d-cite key="comparison"></d-cite>, the size of a dataset played an impact in measuring classification accuracy for NLP tasks. Since the energy dataset is numerical, it's important to test the same concept. We leveraged nearly 150,000 data points, progressively extracting subsets ranging from 10% to 90% of the dataset. For each subset, we trained the architectures, allowing us to explore their performance across varying data volumes.
- **Amount of noise in the dataset**: As discussed in Section 2.2 <d-cite key="trading"></d-cite>, research was done to test LSTMs vs transformers on noisy stock data for various assets. We deemed the energy dataset to be relatively clean since it follows a predictable trend depending on the seasons of the year and time of the day. For example, there are higher energy levels during the winter and daytime hours. To test noise, we added incrementing levels of jittering / Gaussian noise <d-cite key="augmentations"></d-cite> to observe the effect of noisy data on LSTMs and transformers.
- **Output size**: As discussed in Section 2.3

## 4. Experimental Results and Discussion

### 4.1 Size of a Dataset
Given the energy consumption dataset described in Section 3, we trained and evaluated an LSTM model and transformer model on progressively increasing subsets ranging from 10% to 90% of the dataset. The figure below shows the normalized mean squared error (MSE) loss for each subset of the dataset. The experimental results show that transformers have an improving trend   
<p align="center">
  <img src="./assets/img/2023-12-12-time-series-lstm-transformer/lstm_trans_dataset_size_res.png" width="300">
</p>

<p align="center">
  <img src="./assets/img/2023-12-12-time-series-lstm-transformer/test_set_pred_40.png" width="700">
</p>

https://ai.stackexchange.com/questions/20075/why-does-the-transformer-do-better-than-rnn-and-lstm-in-long-range-context-depen this might be helpful for you, the second answer is more concise


### 4.2 Amount of Noise in a Dataset

### 4.3 Output Size

## 5. Conclusion
probably also wanna mention here that one drawback of transformers is that you need a multi-gpu machine in order for it to be parallelizable and train fast. without it, training time is much slower and might not be worth the tradeoffs

<!-- ## 6. References
<p>
<a id="1">[1]</a> 
A. Ezen-Can, “A comparison of lstm and bert for small corpus,” arXiv preprint arXiv:2009.05451, 2020.
</p>
<a id="2">[2]</a> 
P. Bilokon and Y. Qiu, “Transformers versus lstms for electronic trading,” arXiv preprint arXiv:2309.11400, 2023.
<p>
<a id="3">[3]</a>
R. Mulla, "Hourly Energy Consumption," https://www.kaggle.com/datasets/robikscube/hourly-energy-consumption, 2018.
</p>
<p>
<a id="4">[4]</a>
A. Nikitin, "Time Series Augmentations," https://towardsdatascience.com/time-series-augmentations-16237134b29b#:~:text=via%20magnitude%20warping.-,Window%20Warping,-In%20this%20technique, 2019.
</p>
<p>
<a id="5">[5]</a>
G. Ammann, "Using LSTMs And Transformers
To Forecast Short-term Residential
Energy Consumption," https://arno.uvt.nl/show.cgi?fid=160767, 2022.
</p> -->
