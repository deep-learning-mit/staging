---
layout: distill
title: "Tracing the Seeds of Conflict: Advanced Semantic Parsing Techniques for Causality Detection in News Texts"
description: This blog post outlines a research project aiming to uncover cause-effect-relationships in the sphere of (political) conflicts using a frame-semantic parser.
date: 2023-12-12
htmlwidgets: true

# Anonymize when submitting
# authors:
#   - name: Anonymous

authors:
  - name: Philipp Zimmer
    url: "https://www.linkedin.com/in/pzimmer98mit/"
    affiliations:
      name: IDSS, Massachusetts Institute of Technology

# must be the exact same name as your blogpost
bibliography: 2023-11-09-conflict-causality.bib  

# Add a table of contents to your post.
#   - make sure that TOC names match the actual section names
#     for hyperlinks within the post to work correctly.
toc:
  - name: Introduction
  - name: Literature Background
    subsections:
      - name: Qualitative Research on Conflicts
      - name: The Role of Quantitative Methods
      - name: Bridging the Gap with Explainable Modeling Approaches
  - name: Data
    subsections:
      - name: News Articles as Data Source
      - name: Descriptive Analysis of the Data
  - name: Methodology
    subsections:
      - name: The Frame-Semantic Parser
        subsubsections:
          - name: Contextualizing the Approach
          - name: How Does a Frame-Semantic Parser Work?
          - name: Implementation of the Frame-Semantic Parser
      - name: Seed Selection via Semantic Similarity Analysis to Inform Causal Modeling
        subsubsections:
          - name: Understanding Semantic Similarity
          - name: How Do We Compute Semantic Similarity?
      - name: Domain-Specific Metrics
  - name: Findings & Insights
    subsections:
    - name: Frame-Semantic Parser Identifies Causal Frames Reliably
    - name: Differences in Seed Phrase Selection
    - name: Employing Domain-Specific Performance Metrics
  - name: Conclusion & Limitations
    subsections:
      - name: Key Findings
      - name: Limitations & Future Research

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

> *"In the complex world of political conflicts, understanding the underlying dynamics can often feel like trying to solve a puzzle with missing pieces. This project attempts to find those missing pieces through a novel approach that combines the insights of qualitative research with the precision of quantitative analysis."*

{% include figure.html path="/assets/img/2023-11-09-conflict-causality/img1_map.jpg" class="img-fluid" %}
<p align="center" style="color: white; font-style: italic; font-weight: bold;">Retrieved from https://conflictforecast.org</p>

Political conflicts are multifaceted and dynamic, posing significant challenges for researchers attempting to decode their intricate patterns. Traditional methods, while insightful, often grapple with the dual challenges of scale and specificity. This project embarks on an innovative journey to bridge this gap, leveraging a frame-semantic parser to illustrate its applicability for the task and to discuss an approach to achieve domain-specificity for the model using semantic similarity. By synthesizing the depth of qualitative research into the scalability of quantitative methods, we aim to contribute to more informed analyses and actions in low-resource, low-tech domains like conflict studies.

On this journey, the projects key contributions are:

1. **Advancing Frame-Semantic Parsing in Conflict Research**: We introduce the frame-semantic parser, a method that brings a high degree of explainability to conflict studies. Particularly when used in conjunction with news articles, this parser emerges as a powerful tool in areas where data is scarce, enabling deeper insights into the nuances of political conflicts.

2. **Harnessing Semantic Similarity for Domain Attunement**: The project underscores the significance of semantic similarity analysis as a precursor to frame-semantic parsing. This approach finely tunes the parser to specific thematic domains, addressing the gaps often present in domain distribution of common data sources. It illustrates how tailoring the parser input can yield more contextually relevant insights.

3. **Demonstrating Domain-Dependent Performance in Frame-Semantic Parsing**: We delve into the impact of thematic domains on the performance of a transformer-based frame-semantic parser. The research highlights how the parser's effectiveness varies with the domain of analysis, primarily due to biases and structural peculiarities in the training data. This finding is pivotal for understanding the limitations and potential of semantic parsing across different contexts.

4. **Developing Domain-Specific Performance Metrics**: In environments where additional, domain-specific labeled test data is scarce, the project proposes an intuitive method to derive relevant performance metrics. This approach not only aligns the evaluation more closely with the domain of interest but also provides a practical solution for researchers working in resource-constrained settings.

## Literature Background

### Qualitative Research on Conflicts

Qualitative research has long been a cornerstone in the study of political conflicts. This body of work, now well-established, emphasizes the unique nature of each conflict, advocating for a nuanced, context-specific approach to understanding the drivers and dynamics of conflicts. Researchers in this domain have developed a robust understanding of the various pathways that lead to conflicts, highlighting the importance of cultural, historical, and socio-political factors in shaping these trajectories. While rich in detail and depth, this approach often faces challenges in scalability and systematic analysis across diverse conflict scenarios.

### The Role of Quantitative Methods

The emergence of computational tools has spurred a growing interest in quantitative approaches to conflict research. 
These methods primarily focus on predicting the severity and outcomes of ongoing conflicts, with some success <d-cite key="beck2000improving"></d-cite>. However, the onset of conflicts remains challenging to predict, indicating a need for more sophisticated tools and methodologies. Quantitative methods provide scalability and a degree of objectivity but often fail to capture the complexities and evolving nature of conflicts. <d-cite key="goldstein1992conflict"></d-cite>'s work on a conflict-cooperation scale illustrates the difficulty in quantifying conflict dynamics and the controversy in creating aggregate time series from event data. <d-cite key="vesco2022united"></d-cite> highlight the importance of diverse, accurate predictions in conflict forecasting, noting the value of incorporating contextual variables to predict early signals of escalation.

### Bridging the Gap with Explainable Modeling Approaches

The challenge now lies in bridging the insights from qualitative research with the systematic, data-driven approaches of quantitative methods. While the former provides a deep understanding of conflict pathways, the latter offers tools for large-scale analysis and prediction. The key to unlocking this synergy lies in developing advanced computational methods to see the smoke before the fire – identifying the early precursors and subtle indicators of impending conflicts <d-cite key="vesco2022united"></d-cite>. This approach aligns with the evolving needs of conflict research, where traditional models may not adequately address the complex and non-linear nature of conflict data <d-cite key="weidmann2023recent"></d-cite>. <d-cite key="mueller2018reading"></d-cite> demonstrate the potential of utilizing newspaper text for predicting political violence, suggesting a novel data source for uncovering early conflict indicators. However, these early attempts are outdated given the fast technological development in recent years, particularly in the field of natural language processing. This research endeavour seeks to fill that gap and introduce a scalable, explainable method to quantitative conflict research.


## Data

The project capitalizes on the premise that risk factors triggering a conflict, including food crises, are frequently mentioned in on-the-ground news reports before being reflected in traditional risk indicators, which can often be incomplete, delayed, or outdated. 
By harnessing newspaper articles as a key data source, this initiative aims to identify these causal precursors more timely and accurately than conventional methods. 

### News Articles as Data Source

News articles represent a valuable data source, particularly in research domains where timely and detailed information is crucial. In contrast to another "live" data source that currently revels in popularity amongst researchers - social media data - news articles are arguably less prone to unverified narratives. While news articles typically undergo editorial checks and balances, ensuring a certain level of reliability and credibility, they certainly do not withstand all potential biases and are to be handled with caution - as arguably every data source. To counteract potential biases of individual news outputs, accessing a diverse range of news sources is essential. Rather than having to scrape or otherwise collect data on news articles, there is a set of resources available:

* [NewsAPI](https://newsapi.org/): This platform provides convenient access to a daily limit of 100 articles, offering diverse query options. Its integration with a Python library streamlines the process of data retrieval. However, the limitation lies in the relatively small number of data points it offers, potentially restricting the scope of analysis.

* [GDELT Database](https://www.gdeltproject.org/): Renowned for its vast repository of historical information spanning several decades, GDELT stands as a comprehensive data source. Its extensive database is a significant asset, but similar to NewsAPI, it predominantly features article summaries or initial sentences rather than complete texts, which may limit the depth of analysis.

* [Factiva](https://www.dowjones.com/professional/factiva/): A premium service that grants access to the complete bodies of articles from a plethora of global news sources in multiple languages. While offering an exhaustive depth of data, this resource comes with associated costs, which may be a consideration for budget-constrained projects.

* [RealNews](https://paperswithcode.com/dataset/realnews): As a cost-free alternative, this dataset encompasses entire newspaper articles collated between 2016 and 2019. Selected for this project due to its unrestricted accessibility and comprehensive nature, it provides a substantial set of articles, making it a valuable resource for in-depth analysis.

### Descriptive Analysis of the Data

The analysis delved into a selected subset of **120,000 articles** from the [RealNews](https://paperswithcode.com/dataset/realnews) open-source dataset. This subset was chosen randomly to manage the extensive scope of the complete dataset within the project's time constraints. Each article in this subset provided a rich array of information, including **url**, **url_used**, **title**, **text**, **summary**, **authors**, **publish_date**, **domain**, **warc_date**, and **status**.

The range of articles spans from 1869 to 2019, but for focused analysis, we narrowed the scope to articles from **January 2016 through March 2019**. This temporal delimitation resulted in a dataset comprising **58,867 articles**. These articles originated from an expansive pool of **493 distinct news outlets**, offering a broad perspective on global events and narratives. The distribution of these articles across the specified time frame provides the expected observation of increasing news reporting, as visualized below.

{% include figure.html path="/assets/img/2023-11-09-conflict-causality/img4_articlecounts.png" class="img-fluid" %}
<p align="center" style="color: white; font-style: italic; font-weight: bold;">Counts of Articles over Time</p>

To understand the content of our dataset's news articles better, we utilized the *TfidfVectorizer*, a powerful tool that transforms text into a numerical representation, emphasizing key words based on their frequency and distinctiveness within the dataset. To ensure focus on the most relevant terms, we filtered out commonly used English stopwords. The *TfidfVectorizer* then generated a *tf-idf matrix*, assigning weights to words that reflect their importance in the overall dataset. By summing the Inverse Document Frequency (IDF) of each term, we obtained the adjusted frequencies that helped identify the most influential words in our corpus. To visually represent these findings, we created a word cloud (see below), where the size of each word correlates with its relative importance. 

{% include figure.html path="/assets/img/2023-11-09-conflict-causality/img3_wordcloud.png" class="img-fluid" %}
<p align="center" style="color: white; font-style: italic; font-weight: bold;">Word Cloud for Entire News Article Dataset (tf-idf adjusted)</p>

## Methodology

We showcase the applicability of a frame-semantic parsing to the study of conflicts and inform the model with domain-specific seed phrases identified through semantic similarity analysis. This approach not only demonstrates the effectiveness of the method in conflict studies but also showcases how domain-specific applications of deep learning tasks can be accurately applied and measured. Thus, we not only validate the utility of frame-semantic parsing in conflict analysis but also explore innovative ways to tailor and evaluate domain-specific performance metrics.

### The Frame-Semantic Parser

#### Contextualizing the Approach

In the pursuit of bridging the gap between the robust theoretical understanding of conflict dynamics and the practical challenges in data availability, the frame-semantic parser emerges as a promising methodological tool. In a recent study (<d-cite key="balashankar2023predicting"></d-cite>), a team of researchers established a proof-of-concept via its successful application of a frame-semantic parser for the study of food insecurity - a field with similar challenges surrounding data access and quality. While this study relied on what can now be considered the "old state-of-the-art," our proposed approach diverges towards a more contemporary, transformer-based model, inspired by the advancements outlined in <d-cite key="chanin2023open"></d-cite>.

{% include figure.html path="/assets/img/2023-11-09-conflict-causality/img2_parser.png" class="img-fluid" %}
<p align="center" style="color: white; font-style: italic; font-weight: bold;">Retrieved from https://github.com/swabhs/open-sesame</p>


#### How Does a Frame-Semantic Parser Work?

At the heart of frame-semantic parsing, as conceptualized by <d-cite key="gildea2002frame"></d-cite> and formalized by the FrameNet project <d-cite key="baker1998framenet"></d-cite>, is the identification of structured semantic frames and their arguments from natural language text. As illustrated above, these frames encapsulate events, relations, or situations along with their participants, making it a critical tool in natural language understanding (NLU) tasks. The practical applications of frame semantics are broad, ranging from voice assistants and dialog systems <d-cite key="chen2013dialog"></d-cite> to complex text analysis <d-cite key="zhao2023text"></d-cite>.

The process of frame-semantic parsing constitutes three subtasks:

* **Trigger Identification**: This initial step involves pinpointing locations in a sentence that could potentially evoke a frame. It's a foundational task that sets the stage for more detailed analysis.

* **Frame Classification**: Following trigger identification, each potential trigger is analyzed to classify the specific FrameNet frame it references. This task is facilitated by leveraging lexical units (LUs) from FrameNet, which provide a strong indication of potential frames.

* **Argument Extraction**: The final task involves identifying the frame elements and their corresponding arguments within the text. This process adds depth to the frame by fleshing out its components and contextualizing its application within the sentence.

While frame-semantic parsers have arguably not received as much attention as other language modeling methods, three major contributions of the past few years can be highlighted. <d-cite key="swayamdipta2017frame"></d-cite>'s approach - which is still outperforming many other implementations - presented an efficient parser with softmax-margin segmental RNNs and a syntactic scaffold. It demonstrates that syntax, while beneficial, is not a necessity for high-performance frame-semantic parsing. <d-cite key="kalyanpur2020open"></d-cite> explores the application of transformer-based architectures to frame semantic parsing, employing a multi-task learning approach that significantly improves upon previous state-of-the-art results. Most recently, <d-cite key="chanin2023open"></d-cite> developed the first open-source approach - treating frame semantic parsing as a sequence-to-sequence text generation task, utilizing a T5 transformer model. It emphasizes the importance of pretraining on related datasets and employing data augmentations for improved performance. The distinctive strength of a frame-semantic parser lies in its ability to contextualize information, rather than interpreting it in isolation. This feature is particularly invaluable in conflict analysis, where the semantics of discourse play a critical role. 

#### Implementation of the Frame-Semantic Parser

The implementation of our frame-semantic parser involves several key steps. We begin by splitting our text data into sentences using a *split_into_sentences* function. This granular approach allows us to focus on individual narrative elements within the articles and since frame-semantic parsers are reported to perform better on sentence-level <d-cite key="chanin2023open"></d-cite><d-cite key="swayamdipta2017frame"></d-cite>.

In the heart of our methodology, we utilize various functions to extract and filter relevant frames from the text. Our *extract_features* function captures the full text of each frame element, ensuring a comprehensive analysis of the semantic content. The *filter_frames* function then refines this data, focusing on frames that are explicitly relevant to conflict, as informed by research on causal frames in FrameNet.

To optimize the performance of our transformer-based parser, we build a *process_batch* function. This function handles batches of sentences, applying the frame semantic transformer model to detect and filter frames relevant to our study.

Our approach also includes a careful selection of specific frames related to causality and conflict as we are interested in these frames and not just any. We rely on both manually identified frame names (informed by <d-cite key="vieu2016a"></d-cite><d-cite key="vieu2020a"></d-cite>) and pattern-based searches in **FrameNet** to compile a comprehensive list of relevant frames. This curated set of frames is instrumental in identifying the nuanced aspects of conflict narratives within the news articles.

The implementation is designed to be efficient and scalable, processing large batches of sentences and extracting the most relevant semantic frames. This approach enables us to parse and analyze a substantial corpus of news articles, providing a rich dataset for our conflict analysis.

### Seed Selection via Semantic Similarity Analysis to Inform Causal Modeling

#### Understanding Semantic Similarity

Semantic similarity plays a pivotal role in our methodology, serving as the foundation for expanding our understanding of how conflict is discussed in news articles. By exploring the semantic relationships between words and phrases, we can broaden our analysis to include a diverse array of expressions and viewpoints related to conflict. This expansion is not merely linguistic; it delves into the conceptual realms, uncovering varying narratives and perspectives that shape the discourse on conflict.

#### How Do We Compute Semantic Similarity?

To compute semantic similarity and refine our seed phrases, we employ a combination of distance calculation and cosine similarity measures. We begin with a set of initial key phrases **conflict**, **war**, and **battle**, ensuring they capture the core essence of our thematic domain. We then leverage pretrained word embeddings from the *Gensim* library to map these phrases into a high-dimensional semantic space. We also experimented with more sophisticated embedding approaches (like transformer-based) to compute the semantic similarity and thus obtain the seeds. When trading off complexity/time and performance the simpler pretrained *Gensim* model preservered.

Our methodology involves generating candidate seeds from our corpus of documents, including unigrams, bigrams, and trigrams, with a focus on those containing key words related to conflict. We filter these candidates based on their presence in the word vectors vocabulary, ensuring relevance and coherence with our seed phrases.

Using functions like *calculate_distances* and *calculate_cosine_similarity*, we measure the semantic proximity of these candidates to our initial seed phrases. This process involves averaging the distances or similarities across the seed phrases for each candidate, providing a nuanced understanding of their semantic relatedness.

The candidates are then ranked based on their similarity scores, with the top candidates selected for further analysis. This refined set of seed phrases, after manual evaluation and cleaning, forms the basis of our domain-specific analysis, guiding the frame-semantic parsing process towards a more focused and relevant exploration of conflict narratives.

### Domain-Specific Metrics

In the final stage of our methodology, we integrate the identified seed phrases into the frame-semantic parser's analysis. By comparing the model's performance on a general set of sentences versus a subset containing at least one seed phrase, we assess the model's domain-specific efficacy. This comparison not only highlights the general capabilities of large language models (LLMs) but also underscores their potential limitations in domain-specific contexts.

Our approach offers a pragmatic solution for researchers and practitioners in low-resource settings. We demonstrate that while general-purpose LLMs are powerful, they often require fine-tuning for specific domain applications. By utilizing identified domain-specific keywords to construct a tailored test dataset, users can evaluate the suitability of general LLMs for their specific needs.

In cases where technical skills and resources allow, this domain-specific dataset can serve as an invaluable tool for further refining the model through data augmentation and fine-tuning. Our methodology, therefore, not only provides a robust framework for conflict analysis but also lays the groundwork for adaptable and efficient use of advanced NLP tools in various thematic domains.

We present the results for these domain-specific measure for **F1 score**, **recall**, and **precisions**. Likewise, to illustrate performance differences across domains, we conducted the entire approach also for the finance domain, starting with the keywords **finance**, **banking**, and **economy**.


## Findings & Insights

### Frame-Semantic Parser Identifies Causal Frames Reliably

In this stage, we assess if the methodology is truly applicable to the domain of conflicts and for the use with news article data. We find that of our 37 identified cause-effect related frames, all are represented with various instances in our dataset. In fact, as few as 1,600 randomly selected news articles (processed in 100 batches of 16 batch samples) suffice to cover all cause-effect related frames. Therefore, for this intermediate step of the project, we gather support that the parser is in-fact applicable to news article data.

### Differences in Seed Phrase Selection

We make one major observation between the results of the finance- versus conflict-specific seed selection for downstream use. Potentially driven by the fact that conflicts are drastically driven by geographic labels and information, a number of the top 50 seed phrases were geographic terms like "Afghanistan." Since we did not want to bias the downstream evaluation of our domain-specific metrics we excluded these seed phrases and continued the analysis with 34 seeds. In contrast, the top 50 finance-specific seed phrases obtained from the semantic analysis were neither geographic nor linked to individual (financial) historic events, wherefore we continued the downstream analysis with all top 50 seed phrases. Already here we can observe the deviances across domains, given more support to the idea of domain-specific evaluation and metrics.

### Employing Domain-Specific Performance Metrics

Our research involved an extensive evaluation of the frame-semantic parser, based on a transformer architecture, across various configurations and domain-specific datasets. We began by rebuilding and training the model using the vanilla code and a smaller model size without hyperparameter tuning. Subsequently, we fine-tuned the hyperparameters to match the baseline performance levels. After this, we move to one of the main contributions of this project: the domain-specific evaluation. The evaluation was carried out on domain-specific validation and test datasets, curated using seed words from **finance** and **conflict** domains to highlight differences across domains.

The untuned model (*validation n = 646, test n = 1891*) showed an argument extraction **F1 score of 0.669** and a **loss of 0.181** on the validation set. On the test set, it presented a slightly similar **F1 score of 0.669** and a **loss of 0.227**.
Hyperparameter-Tuned Performance

Post hyperparameter tuning, there was a notable improvement in the model's validation performance (*n = 156*), with the **F1 score for frame classification reaching as high as 0.873**, and the **precision for trigger identification at 0.818**. The test metrics (*n = 195*) also showed consistent enhancement, with the **F1 score for frame classification at 0.864** and **trigger identification precision at 0.747**.

When evaluated on domain-specific datasets, **the model exhibited varying degrees of effectiveness** which showcases our assumption that domains matter to the applicability of LLMs to domain-specific tasks and that our simple proposed way of generating domain-specific metrics can give insights on that. For the conflict keywords (*validation n = 121, test n = 255*), the model achieved a **validation F1 score of 0.865 for frame classification and 0.764 for trigger identification precision**. However, for the finance domain (*validation n = 121, test n = 255*), the **F1 score for frame classification was slightly higher at 0.878**, while the **trigger identification precision was lower at 0.781** compared to the conflict domain.

The results indicate that the hyperparameter-tuned model significantly outperforms the vanilla model across all metrics. Additionally, domain-specific tuning appears to have a considerable impact on the model's performance, with the finance domain showing slightly better results in certain metrics compared to the conflict domain. These insights could be pivotal for further refinements and targeted applications of the frame-semantic parser in natural language processing tasks. Moreover, these observation fit our general understanding of the two domains. Reports on conflicts are likely to discuss the involved parties' reasons for specific actions like attacks on certain targets. Additionally, the actions in conflicts are arguably more **triggering** events than "the good old stable economy." Certainly, this research project can only be the beginning of a more rigorous assessment, but these findings show great promise of the idea of **generating and evaluating simple, domain-specific performance metrics**.

{% include figure.html path="/assets/img/2023-11-09-conflict-causality/img5_performance.png" class="img-fluid" %}
<p align="center" style="color: white; font-style: italic; font-weight: bold;">Performance Evaluation of Frame-Semantic Parser</p>

## Conclusion & Limitations

This project has embarked on an innovative journey, merging advanced natural language processing techniques with the intricate study of conflict. By harnessing the power of a transformer-based frame-semantic parser and integrating semantic similarity analysis, we have made significant strides in identifying causal relationships within news articles. This methodology has not only illuminated the dynamics of conflict as portrayed in media but also demonstrated the adaptability and potential of frame-semantic parsing in domain-specific applications.

### Key Findings

1. **Utility of Frame-Semantic Parsing**: Our work has showcased the frame-semantic parser as a valuable and explainable tool, particularly effective in data-scarce environments like conflict research. Its ability to contextualize information and discern nuanced semantic relationships makes it an indispensable asset in understanding complex thematic domains.

2. **Semantic Similarity for Domain-Specific Perspective**: We illustrated the effectiveness of using semantic similarity to refine seed phrases, thereby tailoring the frame-semantic parser to the specific domain of conflict. This approach has proven to be a straightforward yet powerful means to customize advanced NLP models for targeted analysis.

3. **Dependence on Domain for Model Performance**: Our findings highlight a significant insight: the performance of general-purpose language models can vary depending on the domain of application. This observation underscores the need for domain-specific tuning to achieve optimal results in specialized contexts.

4. **Development of Domain-Specific Performance Metrics**: We proposed and validated a practical approach to developing domain-specific metrics, especially useful in resource-constrained environments. This methodology enables a nuanced evaluation of model performance tailored to specific thematic areas.

### Limitations & Future Research

Despite the promising results, our project is not without its limitations, which pave the way for future research opportunities:

1. **Data Dependency**: The effectiveness of our approach is heavily reliant on the quality and diversity of the news article dataset. Biases in media reporting or limitations in the scope of articles can skew the analysis and affect the accuracy of the results. In an extended version of the project - and with funding - one could switch to the [Factiva](https://www.dowjones.com/professional/factiva/) dataset.

2. **Applicability of Domain-Specificity to Other Themes**: While our method has shown efficacy in the context of conflict analysis, its applicability to other specific domains requires further exploration. Future research could test and refine our approach across various thematic areas to assess its broader utility.

3. **Model Complexity and Interpretability**: While we have emphasized the explainability of the frame-semantic parser, the inherent complexity of transformer-based models can pose challenges in terms of scaling and deployment. Future work could focus on simplifying these models without compromising their performance - for instance via pruning and quantization.

4. **Expansion of Semantic Similarity Techniques**: Our semantic similarity analysis was instrumental in refining seed phrases, but there is room for further enhancement. Incorporating more advanced semantic analysis techniques could yield even more precise and relevant seed phrases. While we found alternative methods, like BERT-based approaches to not yield significant improvements, ever more models flood the market.

5. **Integration with Other Data Sources**: Expanding the dataset beyond news articles to include social media, governmental reports, or academic literature could provide a more holistic view of conflict narratives and their causal relations.

In conclusion, our project represents a significant step forward in the intersection of natural language processing and conflict research. By addressing these limitations and building on our foundational work, future research can continue to push the boundaries of what is possible in this exciting and ever-evolving field.