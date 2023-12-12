---
layout: distill
# CHANGE TITLE LATER TO WHATEVER WE PUT ON THE PROPOSAL
title: Modeling Elephantfish Communication through Deep RNNs
description: Elephantfish represent a fascinating subject for study within the realms of bioacoustics and animal communication due to their unique use of electric fields for sensing and interaction. This project proposes the development of a deep learning framework to model the electrical communication signals of elephantfish, akin to language models used in natural language processing (NLP). 
date: 2022-12-01
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
bibliography: 2022-12-01-distill-example.bib  

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

## Discussion and Conclusions

The successful application of LSTM RNNs to model elephantfish communication signals represents a significant step forward in our understanding of non-human communication systems. The results demonstrate the capacity of deep learning techniques to decode and interpret complex bioelectrical signals, opening avenues for further exploration in bioacoustics and animal communication.

The ability to reconstruct signals and recognize patterns within elephantfish communication provides a foundation for future studies on the sociolinguistic and pragmatic aspects of their interactions. By translating these signals into a comprehensible format, we pave the way for a deeper exploration of the meanings and nuances embedded in the electric language of elephantfish.

However, it is crucial to acknowledge the challenges and limitations of this research. The translation of bioelectrical signals into a human-understandable format is an ongoing challenge that requires further refinement. Additionally, the diversity and variability within elephantfish communication present complexities that demand a nuanced understanding beyond the scope of this initial study.

## Challenges and Future Directions

### Challenges

1. **Signal Translation:** Translating electric signals into a human-interpretable format poses a considerable challenge. Developing innovative visualization or auditory representations that convey the richness of these signals is an area that requires ongoing research.

2. **Variable Communication Contexts:** Elephantfish engage in diverse communication contexts, and their signals can vary significantly. Adapting the model to handle this variability and understanding the context-specific nuances remain ongoing challenges.

3. **Interdisciplinary Collaboration:** Bridging the gap between biology, linguistics, and computer science requires effective communication and collaboration. Establishing a common ground for interdisciplinary research is an ongoing process that demands continuous effort.

### Future Directions

1. **Multimodal Representation:** Integrating multiple modalities, such as visual and auditory representations, can enhance the human interpretability of the modeled signals. Exploring ways to combine these modalities is a promising avenue for future research.

2. **Real-time Monitoring:** Extending the application of the model to real-time monitoring of elephantfish communication in their natural habitat presents exciting possibilities. This could involve the development of autonomous monitoring systems that leverage the trained model for on-the-fly interpretation of signals.

3. **Cross-Species Comparative Studies:** Applying similar deep learning approaches to other species with unique communication methods can offer insights into the evolution and diversity of animal communication. Comparative studies across species can reveal common principles and unique adaptations.

4. **Ethological Investigations:** Integrating ethological observations with the deep learning model's outputs can provide a richer understanding of the behavioral implications of different communication patterns. This involves collaboration with experts in animal behavior and ethology.

In conclusion, this research project represents a pioneering effort in utilizing deep learning to unravel the mysteries of non-human communication. The successful application of LSTM RNNs to model elephantfish communication signals opens new doors for interdisciplinary exploration and understanding. As we continue to address challenges and venture into future directions, the study of elephantfish communication stands as a testament to the potential of collaborative and innovative research at the intersection of biology and artificial intelligence.

