# Overview:

Deep learning's influence on computer science is nowhere more evident than in its role in refining document ranking systems critical to information retrieval. Our project, inspired by the "Teach LLMs to Personalize" paper, seeks to push the envelope further by integrating Agent-based Large Language Models (Agent LLMs) into document ranking frameworks. We hypothesize that this integration could significantly boost performance, and our goal is to provide empirical evidence to support or refute this.

To achieve a deeper understanding of deep learning's role in document ranking, we will engage in original analysis and experimentation, with a focus on producing novel insights. Our findings will be distilled into a high-quality, clear blog modeled after distill.pub's exemplary communication standards. Our research will pivot on three primary questions: the impact of Agent LLMs on ranking accuracy, the insights extracted from their behavior, and a comparative analysis with the personalized learning framework proposed in the "Teach LLMs to Personalize" paper.

# Implementation Approach:

Methodologically, we'll reconstruct and adapt the framework from the paper, integrating Agent LLMs in the initial document ranking phase and embarking on a rigorous empirical analysis process, involving data preprocessing and robust system training. We aim to determine whether Agent LLMs enhance or detract from system performance, using metrics likely derived from the "Teach LLMs to Personalize" paper. The outcomes, whether they indicate improvements or drawbacks, will be carefully visualized and interpreted, contributing valuable insights into the behavior of Agent LLMs in document ranking.

Our project will particularly focus on reimplementing the “Ranking” portion of the personalized generation framework (see Figure 1 below), by using fine tuned LLM Agent(s) instead of ranking metrics used in the research paper (RankDocBM25, RankDocDense, RankSnippet, RankDocBySnpt). We intend to utilize the same datasets used in the research paper (CNN/Daily Mail [30], ForumSum [9], and Reddit TIFU-long [10]) to maintain data consistency between the two approaches. We will also attempt to experiment with different, specified fine tunings of the LLM Agent(s) to see if models that focus on different metrics perform better (i.e. fine tune a model to analyze past documents based on stylistic metrics–sentence structure, word choice, etc.–to see if they perform better).

![Image of Personalized Gen Framework](./assets/img/PersonalizationGenFrmwrk.png)

# Timeline:

The project will proceed according to a timeline that includes setting up the experimental framework, data preprocessing, system training, and result analysis. The concluding phase will focus on composing and refining the blog content to ensure it effectively communicates our findings. By having a structured timeline, we expect to contribute meaningfully to the field's understanding of document ranking's which utilize Agent LLMs and their role in creating personalized outputs.
