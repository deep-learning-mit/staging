# 6.S898 Final Project Proposal - LSTM vs Transformers for Time Series Modeling

By Miranda Cai, Roderick Huang

## 1. Introduction

For our final project, we will perform a comparative analysis of LSTMs and transformers in the context of time series forecasting. Traditionally, most models that make time series predictions have relied on LSTM models because of an LSTM's ability to recognize sequence patterns of any length using its long-term memory. While the accuracy of such models have been shown to be quite effective in many applications, training LSTM models takes a relatively long time because of the fact that they must remember all past observances.

One faster alternative to LSTM models are transformers. Transformers are able to remember only the important bits of inputs using an attention-mechanism, and is also parallelizable making it much faster to train than recursive LSTMs that must be processed sequentially. With its recent development, people have started opting to use transformer based models to solve sequence problems that once relied on LSTMs. One significant example is for NLP use cases, where transformers can process sentences as a whole rather than by individual words like LSTMs do. However, since transformers have been around for less than a decade, there are still many potential applications that are yet to be deeply explored.

Thus, we would like to explore the effectiveness of transformers specifically for time series forecasting. Our goal is to realize which particular features of time series datasets could lead transformer-based models to outperform LSTM ones. We plan to evaluate our experiments on both training time and accuracy.

## 2. Investigation and Analysis

### 2.1 Comparative Analysis

To perform a comparative analysis of LSTMs and transformers, we intend to utilize PyTorch to implement an LSTM model and a transformer model that will be both trained on a time-series datasets to pinpoint the advantages and disadvantages of each architecture. We will be comparing the following features for datasets:

- **Small versus Large Datasets**: The size of a dataset should play a role in the performance of an LSTM model versus a transformer model. A study [1] done in the NLP field compared a pre-trained BERT model with a bidirectional LSTM on different language dataset sizes. They experimentally showed that the LSTM accuracy was higher by 16.21\% relative difference with 25\% of the dataset versus 2.25\% relative difference with 80\% of the dataset. This makes sense since BERT is a robust transformer architecture that needs more data. As shown in the figure below from [1], while LSTM outperformed BERT, the accuracy difference gets smaller as the perctange of training data used for training increases. With smaller datasets, it's likely that BERT will overfit. We predict that in time series datasets, a similar pattern should appear where LSTMs work better for smaller datasets and transformers become better for larger datasets.

![Figure 1 - LSTM outperforms BERT for all partitions of a dataset](assets/img/2023-12-12-time-series-lstm-transformer/dataset_size_research_fig.png)

- **Clean versus Noisy Datasets**: Theoretically, LSTMs are more robust to noisy data due to its ability to capture local dependencies. On the other hand, the self-attention mechanisms in transformers propagate errors and may struggle with sequences that have a high degree of noise. Electronic traders have been recently attempting to apply transformer models in financial time series prediction to beat LSTMs [2]. Financial data sets are known to be extremely noisy. Experimental results have shown that transformer models have limited advantage in absolute price sequence prediction. In other scenarios like price difference and price movement, LSTMs had better performance.

Since LSTMs have been around much longer than transformers, they're usually the primary architecture for time series forecasting. However, recently, intense debates have risen after research has shown that transformers can be designed in such a way that they can perform better than LSTMs. The Autoformer architecture [3] adds series decomposition blocks to focus on seasonal patterns which is common in time series datasets.

We hope that in this project, we can pinpoint some features that allow transformer models to potentially outperform LSTM models.

### 2.2 Evaluation Metrics

The combination of architectures and datasets will be evaluated with _efficiency_ and _accuracy_. Efficiency will be measured through the time it takes the model to train a dataset. Accuracy will be measured by the mean squared error (MSE) loss of the test set or future time series data. Another possible measure of accuracy is Mean Absolute Scaled Error (MASE) [4] which is commonly used in evaluating time series forecasting modeling.

### 2.3 Hypothesis

We plan to utilize an energy consumption dataset [5] for our analysis. This choice is driven by the dataset's relative simplicity in terms of data cleaning and its greater accessibility in comparison to financial datasets. By investigating the dataset type and size, we have formulated the following hypotheses.

|               | Small Dataset | Large Dataset |
| ------------- | ------------- | ------------- |
| Clean Dataset | LSTM          | Transformer   |
| Noisy Dataset | LSTM          | ???           |

As depicted in the table, we have a keen interest in assessing whether transformers can surpass LSTM models in performance when confronted with larger and more noise-prone datasets. This combination has been the subject of significant debate and continues to pique the interest of researchers, making it a noteworthy area of investigation based on prior research.

## 3. Timeline

- Week 1 (11/09 - 11/14): Building a basic transformer model and an LSTM model that work to start with.
- Week 2 (11/14 - 11/21): Finding datasets that each meet the different conditions stated above. Primarily making sure our LSTM model is able to produce good results since the LSTM acts as our benchmark.
- Week 3 (11/21 - 11/28): Tuning and evaluating our transformer model on the same datasets to compare. In this process, it's very possible that we find different features of datasets that we think might make a starker difference between transformer and LSTM performance.
- Week 4 (11/28 - 12/05): Analyzing the results of our two models and drawing conclusions from what we have observed.
- Week 5 (12/05 - 12/12): Piecing everything together for the blog, also using this final week as a grace period to resolve any possible issues we might encounter.

## 4. References

[1] A. Ezen-Can, “A comparison of lstm and bert for small corpus,” arXiv preprint arXiv:2009.05451, 2020.
[2] P. Bilokon and Y. Qiu, “Transformers versus lstms for electronic trading,” arXiv preprint arXiv:2309.11400, 2023.
[3] A. Zeng, M.Chen, L. Zhang, and Q. Xu, “Are transformers effective for time series forecasting?,” arXiv preprint arXiv:2205.13504, 2022.
[4] “Metric:mase.”
[5] “Hourly energy consumption.”
