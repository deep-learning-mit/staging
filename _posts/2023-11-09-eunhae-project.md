---
layout: distill
title: Mitigating catastrophic forgetting in LLMs through a dynamic memory bank approach (Project proposal)
description: Can we integrate a dynamic memory bank in LLMs to effectively mitigate catastrophic forgetting?
date: 2023-11-09
htmlwidgets: true

authors:
  - name: Eunhae Lee
    url: "https://www.linkedin.com/in/eunhaelee/"
    affiliations:
      name: MIT

# must be the exact same name as your blogpost
bibliography: 2023-11-09-eunhae-project.bib  

# Add a table of contents to your post.
#   - make sure that TOC names match the actual section names
#     for hyperlinks within the post to work correctly.
toc:
  - name: Abstract
  - name: Proposed outline
  - name: Relevant literature
  - name: Open questions
  # - name: Appendix

---

## Abstract

**Title**: "Mitigating Catastrophic Forgetting in LLMs through a Dynamic Memory Bank Approach" (TBD)

**Research Question**: How can the integration of a dynamic memory bank in LLMs effectively mitigate catastrophic forgetting, particularly when learning sequential or non-stationary data?

**Objective**: To design, implement, and evaluate a proof-of-concept dynamic memory bank mechanism that allows an LLM to preserve and revisit previous knowledge, improving retention without compromising the acquisition of new information.

**Hypothesis**: A memory bank that dynamically updates and retrieves knowledge can significantly reduce catastrophic forgetting in LLMs by serving as an external, durable memory system, akin to a long-term memory in humans.

**Background:** Large language models have shown remarkable capabilities when it comes to producing human-like text and having a two-way conversation. However, catastrophic forgetting — the tendency of LLMs to lose previously acquired information when new information is presented — poses a significant challenge. This phenomenon is particularly problematic in dynamic environments (i.e. when new information is presented, such as user feedback) where continuous learning is essential. For this project, I will explore the concept of a dynamic memory bank to LLMs to help them learn incrementally from user interactions without forgetting previously learned information. I will explore existing approaches to tackle this issue, and propose a new or updated approach to the concept of memory bank.


## Proposed outline

1. **Literature review**: Review existing solutions to catastrophic forgetting, such as elastic weight consolidation (EWC), replay methods, and [dual-memory learning systems](https://arxiv.org/abs/1710.10368). Evaluate their strengths and limitations in the context of LLMs.

2. **Design of memory bank**: Deep dive into the novel design of the [“MemoryBank” (Zhong et al., 2023)](https://arxiv.org/abs/2305.10250) including their design/conceptualization of memory storage, memory retrieval, and memory updating mechanism. Some questions I will ask when deep diving are:
    1. When and how is the memory bank updated?
    2. What are the retrieval mechanisms that allow efficient retrieval of stored knowledge?
    3. How does the model dynamically update its understanding of user personality? 
    4. How does memory updating work in a dynamic environment when new information is constantly being presented?
    5. What is a new approach/angle that can be added to this model?
3. **The new idea and analysis**
    1. Once I review the existing work, I plan to propose a new approach/idea to the concept of memory bank that could be incorporated from insights from cognitive psychology, neurology, or other fields. Specifically, I’m intrigued by the concept of “Theory of Mind” which may be useful in improving the representation of user personality understanding and thus overall performance of the memory bank.
        1. Originating from cognitive and behavioral sciences, "Theory of Mind" refers to the cognitive capacity to attribute mental states to oneself and to others. It is the ability to understand that others have beliefs, desires, intentions, and perspectives that are different from one's own. In the context of human-AI interaction, theory of mind is interesting as it relates to whether and how AI systems can understand and predict human mental states and behaviors.
    2. Measurement of Improvement: Explore how to measure improvements in catastrophic forgetting, such as benchmarks or metrics. 
    3. Experimentation (stretch goal): I could design a small scale experiment to measure the improvement.
4. **Practical applications**: Discuss how improvements in catastrophic forgetting through memory banks could affect the deployment LLMs in real-world scenarios. Could potentially do a deeper dive into one specific use case (i.e. financial robo-advisor)


## Relevant literature

Here are some research papers related to catastrophic forgetting in large language models (LLMs) and the concept of a memory bank or related techniques to mitigate such forgetting, in no particular order:

1. **[Overcoming catastrophic forgetting in neural networks (Kirkpatrick et al., 2017)](https://www.pnas.org/doi/10.1073/pnas.1611835114):** This study proposes an approach that remembers old tasks by selectively slowing down learning on the weights important for those tasks. 
2. **[Deep Generative Dual Memory Network for Continual Learning (Kamra et al., 2017)](https://arxiv.org/abs/1710.10368):** The authors derive inspiration from human memory to develop an architecture that can continuously learn from sequentially incoming tasks, while avoiding catastrophic forgetting. Specifically, they incorporate a dual memory architecture that emulates the complementary learning systems of the human brain.
3. **[An Empirical Study of Catastrophic Forgetting in Large Language Models During Continual Fine-tuning (Luo et al., 2023)](https://arxiv.org/abs/2308.08747)**: This study examines the phenomenon of catastrophic forgetting specifically in the context of LLMs undergoing continual fine-tuning.
4. **[Dynamic Memory to Alleviate Catastrophic Forgetting (Perkonigg et al., 2021)](https://www.nature.com/articles/s41467-021-25858-z)**: The research discussed here adapts models to continuous data streams, employing dynamic memory techniques to perform rehearsal on diverse subsets of training data as a countermeasure to catastrophic forgetting.
5. **[MemoryBank: Enhancing Large Language Models with Long-Term Memory (Zhong et al., 2023)](https://arxiv.org/abs/2305.10250):** This paper deals directly with the concept of enhancing LLMs with a long-term memory mechanism, addressing the deficiencies in current models for applications requiring sustained interaction, which could be integral to resolving issues of catastrophic forgetting.
6. **[Investigating the Catastrophic Forgetting in Multimodal Large Language Models (Zhai et al., 2023)](https://nips.cc/virtual/2023/79641)**: Following the development of GPT-4, this paper explores the occurrence of catastrophic forgetting in multimodal LLMs, emphasizing the challenge as these models are fine-tuned from pre-trained states.
7. **[Interactive AI with a Theory of Mind (Çelikok et al., 2019)](https://arxiv.org/abs/1912.05284)**: The authors formulate human-AI interaction as a multi-agent problem, endowing AI with a computational theory of mind to understand and anticipate the user.


## Open questions

1. Is the research approach of looking into the "MemoryBank" concept based on a very recent paper and expanding it a good direction for this project?
2. How can I scope the project so it's feasible in 4-5 weeks? What are some specific ideas on how I could scope down if needed?
3. Would there be a scenario where I could forego the experimentation but still be able to fulfill the requirements of the project (mostly due to sake of time)? 


<!-- ## Appendix -->

