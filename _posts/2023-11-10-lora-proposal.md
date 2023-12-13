---
layout: distill
title: LoRA proposal
description: This is our project proposal
date: 2023-11-10
htmlwidgets: true

# Anonymize when submitting
# authors:
#   - name: Anonymous

authors:
  - name: Semyon Savkin
    affiliations:
      name: MIT
  - name: Egor Lifar
    affiliations:
      name: MIT

# must be the exact same name as your blogpost
bibliography: 2023-11-10-lora-proposal.bib
---

## Project proposal

Low-rank approximation is a way to compactly store a parameter matrix, and perform fast inference using this matrix. The key idea behind low-rank approximation is to represent an $$N \times M$$ matrix as a product of two matrices with sizes $$N \times K$$ and $$K \times M$$, where K is significantly smaller than N or M. It turns out that many matrices have low-rank approximations that are close to them.

We see two possible ways to utilize low-rank approximation in model training. One idea shows how to use low-rank representation of a matrix in model finetuning. Let A be a parameter matrix of the base model, then we represent a new parameter matrix as $$A + BC$$, where $$BC$$ is a low-rank approximation of the difference in weights. This result has been successful in finetuning large language models <d-cite key="hu2021lora"></d-cite>, or generative text-to-image models <d-cite key="smith2023continual"></d-cite>.

Another idea is to try to distill a model, getting a new model with fewer parameters and comparable performance. For each of the weights of the model, we can use SVD decomposition to get its low-rank representation. Then, we fine-tune the new representations on a dataset, generated from running the original model on various inputs.

In our project, we plan to experiment with both approaches in several domains:

* We can start from classification models. Our goal is to reduce the size of a model by finding an equivalent low-rank representation. The benefit of working with classification tasks is that the metric of success is clear, so it will be easier to identify tradeoffs between performance and compression.

* We can finetune an image classification network to work across different domains using LoRA.

* Then, if we have time, we could apply LoRA to finetuning BeRT for identifying tags in competitive programming problems. We were able to do it by finetuning the whole model, so we could compare the those results with LoRA.

* Another idea is to finetune canny edges control net using LORA <d-cite key="zhang2023adding"></d-cite> for stable diffusion <d-cite key="rombach2022highresolution"></d-cite>, to get a different image conditioning criterion.

* We can think of efficient ways to compose different LoRA conditionings

