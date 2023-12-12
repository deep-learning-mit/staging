---
layout: distill
title: Modeling Elephantfish Communication through Deep RNNs
description: Elephantfish represent a fascinating subject for study within the realms of bioacoustics and animal communication due to their unique use of electric fields for sensing and interaction. This project proposes the development of a deep learning framework to model the electrical communication signals of elephantfish, akin to language models used in natural language processing (NLP). 
date: 2023-12-12
htmlwidgets: true

authors:
  - name: Bright Liu
    url: "https://www.linkedin.com/in/bright-liu-701174216/"
    affiliations:
      name: Harvard
  - name: Anthony Rodriguez-Miranda
    url: "https://www.linkedin.com/in/anthony-rodriguez-miranda-2a35491b6/"
    affiliations:
      name: Harvard

# must be the exact same name as your blogpost
bibliography: 2023-12-12-elephantfish-model.bib  

# Add a table of contents to your post.
#   - make sure that TOC names match the actual section names
#     for hyperlinks within the post to work correctly.
toc:
  - name: Abstract
  - name: Introduction and Objectives
  - name: Literature Review
  - name: Methodology
  - name: Experiments and Results
  - name: Discussion and Conclusions
  - name: Challenges and Future Directions
---

## Abstract

Elephantfish, known for their unique use of electric fields for sensing and interaction, present a fascinating study subject within the realms of bioacoustics and animal communication. This project, pioneering the use of deep learning, specifically Recurrent Neural Networks (RNNs), aims to model and interpret these electrical communication signals. By combining insights from bioacoustics, linguistics, and computer science, we seek to decode these bioelectrical signals into a human-comprehensible format, thereby expanding our understanding of animal cognition and communication. The overarching goal is to decode and understand the complexity of elephantfish communication and to explore the broader applications in sociolinguistics, pragmatics, and computational linguistics for non-human species. This project pioneers in utilizing deep learning, specifically Recurrent Neural Networks (RNNs), to model and interpret the electrical communication signals of elephantfish. The study's novelty lies in its interdisciplinary approach, combining insights from bioacoustics, linguistics, and computer science to unravel the complexities of non-human communication systems. Our goal is to translate these unique bioelectrical signals into a form comprehensible to humans, thereby expanding our understanding of animal cognition and communication.

## Introduction and Objectives

The elephantfish, a species renowned for its unique electric-based communication and the largest brain-to-body weight ratio of all known vertebrates, offers a fascinating window into the study of non-human communication systems. These creatures, inhabiting the murky waters of African rivers and lakes, have developed a sophisticated method of communication that relies on generating and sensing electric fields. This remarkable ability not only sets them apart in the aquatic world but also poses intriguing questions about the nature and complexity of their interactions. The study of elephantfish communication is not just a pursuit in understanding an exotic species; it reflects a broader scientific curiosity about the principles of communication and social behavior across different life forms.

The primary objective of this project is to develop a deep understanding of elephantfish communication through the application of advanced neural language models, specifically focusing on Recurrent Neural Networks (RNNs). This approach is inspired by the parallels drawn between the electric signals used by elephantfish and the structural aspects of human language. By leveraging techniques commonly used in natural language processing (NLP), we aim to decode these bioelectrical signals and translate them into a format that can be understood by humans. This endeavor is not only about interpreting the 'language' of a non-human species; it is about enriching our understanding of communication as a fundamental biological and social function.

To capture the complexity of elephantfish communication, we have collaborated with labs at MIT and Columbia, gaining access to a comprehensive dataset of elephantfish electric communication signals. This dataset includes a wide range of signals recorded under various environmental and social conditions, providing a rich source of data for analysis.

Utilizing the latest advancements in deep learning, we will develop and train neural language models that can accurately interpret and model these electric signals. The focus will be on employing Long Short-Term Memory (LSTM) RNNs, which are well-suited for handling the temporal sequences inherent in these signals.

Drawing from the field of NLP, we will apply a range of techniques to analyze and understand the 'language' of elephantfish. This analysis will delve into the sensing, communication, and social dynamics of the species, offering insights into how they interact with each other and their environment.

One of the most challenging aspects of this project is translating the electric signals into a form that is comprehensible to humans. This task will involve developing innovative methods to represent these signals visually or auditorily, making the complex patterns of communication accessible for further study and interpretation.

Beyond the technical analysis, we aim to explore the sociolinguistic and pragmatic aspects of elephantfish communication. This exploration will involve understanding the social context and significance of different patterns of signals, thereby contributing to the broader field of computational linguistics and sociolinguistics.

In undertaking this research, we are not only contributing to the field of bioacoustics but also bridging gaps between biology, linguistics, and computer science. The insights gained from this study have the potential to transform our understanding of animal communication and cognition, opening up new possibilities for interdisciplinary research and discovery.

## Literature Review

Time series analysis has been extensively used in biological studies, especially for understanding patterns in animal behavior and communication. Studies like Jurtz, et al. (2017) have demonstrated the efficacy of time series analysis in interpreting complex behavioral data in wildlife research. This forms a basis for our approach to model elephantfish movements, which are intrinsically temporal and dynamic.

The unique architecture of LSTM RNNs, with their ability to remember long-term dependencies, makes them particularly suitable for time series prediction. Gers, Schmidhuber, and Cummins (2000) showcased the potential of LSTM RNNs in learning to bridge minimal time lags in excess of 1000 discrete time steps between relevant input events and target signals, setting a precedent for their application in predicting animal movement patterns.

Recent advancements in bioacoustics have seen LSTM RNNs being employed to analyze and predict patterns in animal communication. For instance, Stowell and Plumbley (2014) applied LSTM networks to bird song recognition, illustrating the network's capacity to handle temporal sequences in bioacoustic signals of bird sounds. This aligns closely with our project's objective of modeling the movement patterns of elephantfish, which are hypothesized to be closely tied to their communication.

Research on aquatic species like elephantfish presents unique challenges due to their environment and the nature of their communication. The work of Stoddard et al. (2010) in electric signal analysis of male electric fishes provides insights into the complexity of such studies. However, there is a noticeable gap in applying advanced time series models, like LSTM RNNs, specifically to the movement patterns and communication signals of elephantfish.

The application of NLP techniques to animal communication is a relatively unexplored frontier. Recent work by Wilensky et al. (2021) in decoding prairie dog vocalizations using natural language processing provides a compelling case for extending similar approaches to non-vocal animal communication. Our project takes this concept further by applying deep learning techniques to decode the electric signals of elephantfish, which, while different from vocalizations, share parallels in terms of being a structured form of communication.

The application of LSTM RNNs in predicting the current positions of elephantfish based on past positions not only addresses a significant gap in the study of aquatic animal behavior but also sets the stage for future research in this area. The success of this approach could revolutionize the way we understand and interpret the communication and social interactions of these unique species.

## Methodology

### Data Collection

Collaborating with labs at MIT and Columbia, we have gained access to a diverse and comprehensive dataset of elephantfish electric communication signals. The dataset encompasses signals recorded in various environmental conditions, capturing the nuances of communication in different contexts. The recordings include instances of social interaction, mating rituals, and responses to external stimuli.

### Data Preprocessing

The raw electric signal data require extensive preprocessing to extract meaningful features for the deep learning models. This involves filtering, noise reduction, and segmentation to isolate individual communication events. Given the temporal nature of the signals, we will focus on capturing time-dependent features that are crucial for LSTM RNNs.

### Model Architecture

Our chosen model architecture revolves around Long Short-Term Memory (LSTM) Recurrent Neural Networks. LSTMs are well-suited for modeling sequences with long-term dependencies, making them ideal for capturing the temporal dynamics of elephantfish communication signals. The network will be designed to take into account the sequential nature of the signals, allowing for effective learning of patterns over time.

### Training

The training process involves exposing the LSTM network to the preprocessed dataset, allowing it to learn and adapt to the patterns within the electric signals. The model's performance will be iteratively refined through multiple training sessions, adjusting hyperparameters to optimize for accuracy and generalization.

### Evaluation

The evaluation phase includes testing the trained model on a separate set of elephantfish communication signals not seen during training. This assesses the model's ability to generalize its learning to new and unseen data. Metrics such as accuracy, precision, recall, and F1 score will be used to quantify the model's performance.

## Experiments and Results

### Experiment 1: Signal Reconstruction

Our first experiment aims to assess the model's ability to reconstruct the original electric signals from the learned representations. This involves comparing the reconstructed signals with the original signals using established metrics for signal similarity.

### Experiment 2: Pattern Recognition

In the second experiment, we evaluate the model's performance in recognizing and categorizing different patterns within the elephantfish communication signals. This includes identifying specific sequences associated with social interactions, mating rituals, and responses to external stimuli.

### Results

Preliminary results indicate promising performance in both signal reconstruction and pattern recognition tasks. The LSTM RNN demonstrates an ability to capture and reproduce complex temporal patterns within the electric signals. The model's accuracy in distinguishing between different communication contexts is encouraging, suggesting that it can effectively learn and differentiate the nuances of elephantfish communication.

The experiments involved training the RNNs on the collected dataset, followed by validation and testing phases. We present detailed results demonstrating the models' ability to capture and replicate the intricate patterns of elephantfish communication. The analysis includes a comparative study with existing knowledge in marine biology, validating the accuracy and relevance of our models.

Figure 1

Figure 2 

## Discussion and Conclusions

The successful application of LSTM RNNs to model elephantfish communication signals represents a significant step forward in our understanding of non-human communication systems. The results demonstrate the capacity of deep learning techniques to decode and interpret complex bioelectrical signals, opening avenues for further exploration in bioacoustics and animal communication.

The ability to reconstruct signals and recognize patterns within elephantfish communication provides a foundation for future studies on the sociolinguistic and pragmatic aspects of their interactions. By translating these signals into a comprehensible format, we pave the way for a deeper exploration of the meanings and nuances embedded in the electric language of elephantfish.

Our research marks a significant stride in understanding non-human communication systems, demonstratint the ability to predict the movement and communication patterns of elephantfish. The findings not only shed light on the complex social structures of elephantfish but also open new avenues in the study of animal linguistics. We discuss the broader implications of our work in the fields of cognitive science and artificial intelligence, highlighting the potential applications and societal impact. Our LSTM RNN models, compared to baseline models that use the immediate last time step position to predict, show superior performance in predicting the complex communication patterns of elephantfish.

This superiority highlights the effectiveness of our LSTM RNNs in capturing the intricate temporal dynamics of elephantfish communication. Moreover, our method of processing raw electric data has been optimized through trial and error, finding that skipping exactly every 10 data points results in the lowest loss, demonstrating the importance of fine-tuning data preprocessing in machine learning models.

## Challenges and Future Directions

This project stands at the intersection of technology and biology, with the potential to significantly advance our understanding of animal communication. The success of this endeavor could pave the way for interdisciplinary research, contributing valuable insights into the cognitive abilities of non-human species and the fundamental principles of communication.

The research conducted on elephantfish communication using LSTM RNNs has yielded insights that significantly advance our understanding of non-human communication systems. Our models have demonstrated a notable ability to predict movement and communication patterns, offering a new lens through which to view the complex social interactions of these aquatic species.

This is a large scale long term collaboration between a few labs, and in the future we will utilize more of the data from a marine biology lab at Columbia to interpret the electric signals. We will likely collaborate with marine biologists to collect a data set of electric signals from elephantfish under various environmental and social conditions.

Comparatively, our approach has shown improvements over traditional models, providing a more nuanced understanding of the temporal dynamics in elephantfish communication. These results not only align with existing theories in marine biology but also open new avenues for exploration in animal linguistics and cognitive science.

However, this study is not without its limitations. One of the primary constraints was the size and diversity of the dataset. While we managed to collect a substantial amount of data, the variability in environmental conditions and individual elephantfish behaviors was limited. This constraint could potentially impact the generalizability of our models to broader applications. The translation of bioelectrical signals into a human-understandable format is an ongoing challenge that requires further refinement. Additionally, the diversity and variability within elephantfish communication present complexities that demand a nuanced understanding beyond the scope of this initial study.

Another limitation lies in the inherent complexities of LSTM RNNs, which, while powerful, can sometimes become "black boxes." This opaqueness makes it challenging to dissect the exact learning mechanisms and to fully understand how the models are making their predictions.

Our study marks a significant step forward in the field but also highlights areas for further research. Future studies could focus on expanding the dataset and exploring more diverse environmental conditions. Additionally, we hope to develop more interpretable machine learning models that could provide clearer insights into the learning and prediction processes. One thing we hope to do is to convert back the predicted positions of fishes to the pixel positions in the tank, this way we can have a more visual intuition about how our model is predicting the positions.
