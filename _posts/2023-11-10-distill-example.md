---
layout: distill
title: Gradient-Boosted Neural Wavlet Interpolation for Time Series (G-BiTS)
description: Your blog post's abstract.
  This is an example of a distill-style blog post and the main elements it supports.
date: 2023-12-12
htmlwidgets: true

authors:
  - name: Yeabsira Moges
    url: "https://www.linkedin.com/in/yeabsira-moges/"
    affiliations:
      name: AI-DS, MIT

# must be the exact same name as your blogpost
bibliography: 2023-11-10-distill-example.bib  

# Add a table of contents to your post.
#   - make sure that TOC names match the actual section names
#     for hyperlinks within the post to work correctly.
toc:
  - name: Introduction
  - name: Related Works
  - name: G-BiTS
  - name: Results
  - name: Analysis
  - name: Conclusions

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

Energy companies struggle with energy allocation. The power grid contains a multitude of homes, schools, and offices all which require different amounts of power draw and capacity. As the current grid stands, the control loop is running on old data and isnt adequately reactive to sudden spikes, as well the inability to properly model trends. Energy forecasting is the means by which we work to rectify that gap. Energy forcasting is a blanket umbrella term coming from general forcasting of any time series data. There are a lot of methods currently available, ranging from purely statistical models up to deep neural networks. At the moment, the SOTA in predictive modeling from statistical models is SARIMAX: Seasonal Autoregressive Integrated Moving Average Exogenous. In deep learning, the SOTA is N-HiTS[1]. Both work well in most circumstances, but there is a lot of work to improve upon the current performance given we want to generate better embeddings to decrease loss through the energy grid. There has been great performance boosts associated with combinding the strengths of the different methods, and that is part of what this paper explores. Another big target: as it stands the current flavors of N-HiTS dont touch upon the further work reccomendations from the original paper. This includes advanced interpolation, moving away from the current linear interpolation for the Block modules and moving towards incorporating wavelet decomposition and transforms to help convert the signal into a form that makes it much easier to deliver robust data. I propose gradient-boosted neural wavlet interpolation for time series (G-BiTS) as a new entry to forcasting models relying on a mix of statistical and neural network based models. G-BiTS expands upon N-HiTS which stands for neural basis expansion analysis for interpretable time series. N-HiTS decompose time series into a set of basis functions, capturing and interpreting temporal patterns. This paper explores ensembling methods and time series analysis.

## Related Works

The main related works relate to the following topics: SARIMAX, N-HiTS, and GBM. SARIMAX stands for seasonal autoRegressive integrated moving average with exogenous variables model. Each element of the SARIMAX are all important in the following ways. AutoRegressive: captures the relationship between an observations at various lags. Integrated: the differencing of raw observations to make the time series stationary. Moving Average: the relationship between an observation and a residual error from a moving average model applied to lagged observations. Seasonal: accounts for seasonality in data, like weekly, monthly, or yearly patterns. Exogenous Variables: These are external variables or predictors that aren't part of the time series itself but are believed to have an impact on it. This is mainly represented in time series analysis by date information with respect to variables unrelated to the power, but can be used to model a common behavior. The biggest flaw with SARIMAX comes from its inability to model more than one seasonality, hampering predictions. A more robust model is N-HiTS which stands for neural basis expansion analysis for interpretable time series forecasting. The best benefit from N-HiTS comes from its ability to learn rich embeddings for time series that properly represent all of the trends and seasonalities inherent to the data, while also producing gains through being able to apply much more data as it is made for longer range predictions. N-HiTS is good, and this paper will be exploring a multiforld extension using gradient boosting [2] and adaptive ensembling[3]. Gradient boosting generates good predictions by training decision trees sequentially. A new tree is modeled on the residual errors made by the preceding trees. Finally, tying everything all together we have wavelet transforms. Wavelets are wave-like oscillations that represent data at various scales effectively. GBMs help us take advantage of a repeated pattern of smooth behavior interrupted by sudden changes or transients in time series data. 

## G-BiTS

This paper proposes a new deep learning framework powered by gradient boosting and signal pre-processing G-BiTS. G-BiTS stands for Gradient-Boosted Neural Wavlet Interpolation for Time Series. G-BiTS builds upon the success of N-HiTS and explores a question posed by the authors in the original paper on replacing the existant sequential projections from the interpolation functions onto wavelet induced spaces, getting high resolution output. G-BiTS is an ensemble model, which is where gradient boosting comes in. The maximum of the combined predictions is taken for adaptive ensembling and higher performance as well as generatily. Max can be min or mean, just depends on the use case and having higher output in this circumstance if perfered. The hope is to use the hourly modeling capabilities of light gradient boosting machines with the versatility of N-HiTS to create a robust ensemble model.

## Results

The testing for the comparisions of the different forcasting methods is based on the BuildingsBench dataset. Specifically, this paper surveys office buildings withing the Fox subsection from the original input. The data includes buildings with energy data that has multiple seasonalities, mostly hourly, daily, weekly, and monthly. Looking at the data, there are some interesting patterns. These are the average skew and kurtosis values for the data: high skew and kurtosis.

Skewness: 1.1118040201238155 
Kurtosis: 3.452262511716185

Statistical analysis also shows that the data was not drawn from a normal ditribution and is not stationary, so the variance and mean were not constant throughout the time series.

Our baseline is simply copying over the values from the previous week and repeating the same for the following week. Non-baseline models tested include the previously mentioned SARIMAX, N-HiTS, LGBM, and G-BiTS. The following are the respective errors from each building ordered as mean average error, root mean squared error, and mean average percent error.

### Building ID: Margarita

SARIMAX (211.47498604910714, 249.84373502456708, 11.805270962305448)

NHITS (21.72069293617509, 27.65604571924576, 1.6335940075280377)

LGBM (33.16067034334621, 41.84784011583212, 2.0058567433490087)

GBITS (26.955107763269822, 31.504577778268615, 1.6841760555882481)

### Building ID: Loreta

SARIMAX (2966.2653087797617, 3513.45974924458, 12.756417057832824)

NHITS (203.50202658318491, 338.92442661325015, 1.0121962487927345)

LGBM (419.71931531784384, 476.48902925976694, 1.8085151798175159)

GBITS (215.94950733822594, 264.7384239183662, 0.9401638424018465)

### Building ID: Gaylord

SARIMAX (1220.2237444196428, 1479.439585459469, 8.095511476323951)

NHITS (137.39752238818102, 203.64435240098928, 0.8720707702102791)

LGBM (347.0178199198448, 435.19043719851146, 2.3137853719619144)

GBITS (21.02548764010548, 27.84334532157823, .73338746467575437)

## Analysis

Across the board, SARIMAX perfofmed the worst, followed closely by NHiTS and LGBMs. The biggest issue with SARIMAX is that it can only take a very limited amount of data, as well as being unable to model multiple seasonalities. G-BiTS showed good adaptability as one model over the large dataset was able to get transferable and adaptible embeddings. The wavelet transforms showed the greatest gains from the interpolation stage as the two level smoothing helped the N-HiTS model better fit the unstationary data. N-HiTS as expected performs well across the board too and had the best time modeling the data.

## Conclusions

There is more work to be done to extend this research topic. Mainly, finding better wavelet decompositions and symmetric recompositions for modeling multiple seasonalities faster and in a more efficient manner. The decomposition showed the biggest gain and confirms the original papers thoughts about the approach. Boosting helped standardize the model and generated really interesting embeddings through the initial wavelet based N-HiTS.

## Bibliography

[1]

N-HiTS: Neural Hierarchical Interpolation for Time Series Forecasting

Cristian Challu, Kin G. Olivares, Boris N. Oreshkin, Federico Garza, Max Mergenthaler-Canseco, Artur Dubrawski

https://arxiv.org/abs/2201.12886

[2]

Gradient Boosting Neural Networks: GrowNet

Sarkhan Badirli, Xuanqing Liu, Zhengming Xing, Avradeep Bhowmik, Khoa Doan, Sathiya S. Keerthi

https://arxiv.org/abs/2002.07971

[3]

Adaptive Ensemble Learning: Boosting Model Performance through Intelligent Feature Fusion in Deep Neural Networks

Neelesh Mungoli

https://arxiv.org/abs/2304.02653
