---
layout: distill
title: "Guided Transfer Learning and Learning How to Learn: When Is It Helpful? (Project Proposal/Outline)"
description: For downstream tasks that involve extreme few-shot learning, it's often not enough to predispose a model 
  with only general knowledge using traditional pre-training. In this blog, we explore the nuances of 
  Guided Transfer Learning, a meta-learning approach that allows a model to learn inductive biases
  on top of general knowledge during pre-training.
date: 2023-11-02
htmlwidgets: true

# Anonymize when submitting
# authors:
#   - name: Anonymous

authors:
  - name: Kevin Li
    url:
    affiliations:
      name: MIT

# must be the exact same name as your blogpost
bibliography: 2023-11-02-guided-transfer-learning.bib  

# Add a table of contents to your post.
#   - make sure that TOC names match the actual section names
#     for hyperlinks within the post to work correctly.
toc:
  - name: "Introduction: Never Enough Data"
    subsections:
    - name: Machine vs Human Intelligence
    - name: "Traditional Transfer Learning: Learning General Knowledge"
  - name: "Guided Transfer Learning and Meta-learning: Learning Inductive Biases"
    subsections:
    - name: Scouting
    - name: Guide Values
    - name: Example Application
  - name: "How Should We Design Scouting Problems?: An Exploration"
    subsections:
    - name: "Consideration 1: Similarity to Downstream Task"
    - name: "Consideration 2: Ease of Convergence"
  - name: "Is GTL Helpful in All Contexts?: An Exploration"
    subsections:
    - name: "Unsupervised Pre-training, Supervised Fine-tuning"
    - name: "Sophisticated Architectures With Built-in Inductive Biases"

---

**[PROJECT PROPOSAL NOTE]**{: style="color: red; opacity: 0.80;" }: In this blog, I'll be discussing and exploring the nuances of a meta-learning pre-training approach known as Guided Transfer Learning (GTL) <d-cite key="gtl"></d-cite>, developed by [Robots Go Mental](https://robotsgomental.com), that improves few-shot learning performance on downstream tasks. I'll begin by motivating and introducting the approach used in the original paper. In fact, I've already drafted the introduction, problem motivation, and the basic outline of an explanation of GTL below. 

After the motivation/high-level introduction, the remainder of the blog has as NOT been drafted yet, but the sections have been outlined below. These sections are just filled with tentative high-level plans for now (which are preceded by a tag like **[TENTATIVE IDEA]**{: style="color: red; opacity: 0.80;" }). In these sections:
-  I'll be going beyond the original GTL paper and exploring some of the nuances of using GTL to effectively predispose models for downstream few-shot learning tasks, with a focus on designing good scouting problems (explained below). This is based on *my own* practical experience of playing with GTL, and was **NOT** discussed in the original GTL paper. I'll create and include some of my own experiements and results to demonstrate my points. 
- I'll also be exploring how GTL can be adapted to and how it performs in various contexts that were **NOT** epxlored in the original GTL paper, with a focus on self-supervised contexts and complex architectures. Again, I'll be creating *my own* experiements to demonstrate the effectiveness/ineffectiveness/challenges of GTL in these contexts.
</span>

## Introduction/Motivation: Never Enough Data

If we take a step back and reflect upon the current state of AI, especially in domains like computer vision and NLP, it appears that the gap between machine and humman intelligence is rapidly narrowing. In fact, if we only consider aspects such as predictive accuracy of discriminatory models and the sensibility of outputs by generative models, it may seem that this gap is almost trivial or even nonexistent for many tasks. However, every time we execute a training script and leave for the next few hours (or few weeks), it becomes abundantly clear that AI is still nowhere near human intelligence because of one critical kryptonite: the amount of data needed to effectively train AI models, especially deep learning models.

While we have tons of training data in domains such as general computer vision (e.g. ImageNet) and NLP (e.g. the entirety of the internet), other domains may not have this luxury. For example, bulk RNA-sequencing data in biomedical research is notoriously cursed with high dimensionality and extremely low sample size. Training AI models on bulk RNA-sequencing datasets often leads to severe overfitting. In order to successfully utilize AI in domains like biomedicine, the highest priority challenge that must be addressed is the one of overcoming the necessity of exuberant amounts of training data. 

### Machine vs Human Intelligence

It often feels like the requirement of having abundant training samples has been accepted as an inevitable, undeniable truth in the AI community. But one visit to a pre-school classroom is all that it takes to make you question why AI models need so much data. A human baby can learn the difference between a cat and a dog after being shown one or two examples of each, and will generally be able to identify those animals in various orientations, colors, contexts, etc. for the rest of its life. Imagine how much more pre-school teachers would have to be paid if you needed to show toddlers thousands of examples in various orientations and augmentations just for them to learn what giraffe is.

Fortunately, humans are very proficient and few-shot learning-- being able to learn from few samples. Why isn’t AI at this level yet? Well, biological brains are not born as empty slates of neurons with random initial connections. Millions of years of evolution have resulted in us being born with brains that are already predisposed to learn certain domains of tasks very quickly, such as image recognition and language acquisition tasks. In these domains, learning a specific task like differntiating between a cat and a dog or between letters of the English alphabet doesn’t require exposure to many samples. Additionally, as we gain more experiences throughout life, we acquire general knowledge that can help us learn new tasks more efficiently if they’re similar to something we’ve learned before. Thus, naturally, the first step toward bridging the gap between natural and machine intelligence is somehow finding a way to predispose an AI to be able to learn any *specific* task within a certain domain with very few samples. The advent of traditional transfer learning has attempted to approach this predisposition task from the "general knowledge" perspective.

### Traditional Transfer Learning: Learning General Knowledge

Transfer learning has been invaluable to almost all endeavors in modern deep learning. One of the most common solutions for tasks that have too little training data is to first pre-train the model on a large general dataset in the same domain, then finetuning the pre-trained model to the more specific downstream task. For example, if we need to train a neural network to determine whether or not a patient has a rare type of cancer based on an X-ray image, we likely will not have enough data to effectively train such a model from scratch. We can, however, start with a model pre-trained on a large image dataset that's not specific to cancer (e.g. ImageNet), and if we *start* with these pre-trained weights, the downstream cancer diagnostic task becomes much easier for the neural network to learn despite the small dataset size.

One way to intuitvely understand why this is the case is through the lens of "general knowledge." When the model is pre-trained on ImageNet data, it learns a lot of knowledge about image data *in general*; for example, the earlier layers of the model will learn low-level features detectors (e.g. edge detectors, simple shape detectors, etc.) that will likely be useful for *any* specific computer vision task. This can be viewed as the model learning "general knowledge" about the domain of image data. When we then fine-tune this model on a cancer dataset, the model doesn't have to relearn the ability to detect these general, low-level features. This general knowledge encoded in the pre-trained weights regularizes the model and mitigates overfitting, as it *predisposes* the model to learn relationships/feature detectors that are generalizable and sensible within the context of image data.

However, if transfer learning could solve all our problems, this blog post wouldn't exist. When our downstream dataset is in the extremeties of the high dimensional, low sample size characterization (e.g. in fields like space biology research, since not many organisms have been to space), learning general knowledge in the form of pre-trained weights isn't enough. How, then, can we predispose models such that they can do extreme few-shot learning, or even *one-shot* learning? Enter guided transfer learning.

## Guided Transfer Learning and Meta-learning: Learning Inductive Biases

Guided transfer learning (GTL) <d-cite key="gtl"></d-cite> is a meta-learning paradigm proposed by the group [Robots Go Mental](https://robotsgomental.com). The main idea for guided transfer learning is that, instead of just having the AI model learn general knowledge, we also want the AI to learn *how* to learn. Specifically, we want it to learn how to pick up new knowledge *most efficiently* for a particular domain, which is RNA-seq data in our case. This means during pretraining, the model, in addition to learning good initial weights, will also learn ***inductive biases*** that affect future training. 

Inductive biases, which affect what kind of functions a model can learn, are usually built into the choice of deep learning arcthiecture, or decided by other hyperparameters we humans choose. With guided transfer learning, they can now be *learned* automatically during pre-training. It’s almost like the model is figuring out some of its own optimal hyperparameters for learning in a particular domain. 

**[TENETATIVE PLAN FOR THE REST OF THE BLOG]**{: style="color: red; opacity: 0.80;" }: In this blog, I'll begin by providing a brief overview of the GTL method (in the following subections of this section) as described in the original GTL, as well as present some cool results from the paper to demonstrate its effectiveness. Then, in the next section, I'll be going beyond the original paper and exploring some of the nuances of using GTL to effectively predispose models for downstream few-shot learning tasks, with a focus on designing good scouting problems. This is based on my own practical experience of using GTL, and was not discussed in the original GTL paper. Finally, in the last section, I'll also be exploring how GTL can be adapted to and how it performs in various contexts that were NOT epxlored in the original GTL paper, with a focus on self-supervised contexts and complex architectures.

### Scouting

**[THE CONTENT IN THIS SECTION IS A TENTATIVE BASELINE]**{: style="color: red; opacity: 0.80;" }

Sounds like magic, right? How does GTL allow a model to *learn* inductive biases? Well, the core behind the GTL approach is a process known as **scouting**, which is an alternative to traditional pre-training. The high-level idea is that it trains copies of the model, called scouts, on easier subproblems. These subproblems should be similar to the target downstream tasks, but easier so that the scouts are more likely to converge. 

In the process of converging, the scouts keep track of what parameters in the model are important to keep flexible for efficient convergence and what parts aren’t. They’re basically logging their learning process. 

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/2023-11-02-guided-transfer-learning/scouting.png" class="img-fluid rounded z-depth-1" %}
    </div>
</div>

For example, if weight `A` increases drastically during training, it’s probably an important weight to change and we should keep it flexible. On the other hand, if weight `B` doesn’t change much at all or fluctuates in a very noisy manner, it is probably not as important to change.  

After the scouts are finished training, the collective feedback from all the scouts is used to decide what inductive biases to impose on the main model such that it can learn most efficiently for the particular domain of data and avoid wasting effort on changing things that don’t really help.


### Guide Values

**[THE CONTENT IN THIS SECTION IS A TENTATIVE BASELINE]**{: style="color: red; opacity: 0.80;" }

So what do these "inductive biases" actually look like, and how do they affect future training? The inductive biases in the context of GTL come in the form of **guide values**. So after scouting, each parameter will not only have its usual weight value, but it will also have a guide value. During gradient decent, the normal update for a particular weight is then multiplied by its corresponding guide value. Thus, the larger the guide value, the more that parameter is allowed to change during downstream training. 

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/2023-11-02-guided-transfer-learning/guide_values_1.png" class="img-fluid rounded z-depth-1" %}
    </div>
</div>

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/2023-11-02-guided-transfer-learning/guide_values_2.png" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    In this very simple neural network with two weights, we can see here that weight `A` has a guide value of 0.56, while weight `B` has a guide value of merely 0.01. Thus, weight `A` is more flexible, in other words allowed to change more, than weight `B` during downstream training. In fact, weight `B` is pretty much frozen, as its guide value of 0.01 makes it so that weight B can barely change throughout training. 
</div>

The goal of scouting is thus to find these optimal guide values, and thus make the *training* process more sparse (i.e. so that only the weights that are useful to change get changed). Note that this is different from making the *neural network* more sparse (i.e. setting weights/connections that are useless to zero).

It’s really quite an elegant and simple approach, the devil is in the details on how to design the subproblems for scouts and how to aggregate the information from scouts to obtain these guide values. 

**[INSERT MATH DETAILS ABOUT HOW GUIDE VALUES ARE CALCULATED AFTER SCOUTING]**{: style="color: red; opacity: 0.80;" }


### Example Application

**[INSERT PROMISING RESULTS FROM EXAMPLE IN ORIGINAL GTL PAPER]**{: style="color: red; opacity: 0.80;" }

## How Should We Design Scouting Problems?: An Exploration

**[TENTATIVE PLAN FOR THIS SECTION]**{: style="color: red; opacity: 0.80;" }: Here, I'll be going beyond the original paper and exploring some of the nuances of using GTL to effectively predispose models for downstream few-shot learning tasks, with a focus on designing good scouting problems. This is based on my own practical experience of using GTL, and was not discussed in the original GTL paper. I'll be focusing on the BALANCE between two important considerations when deciding the scouting task. I'll be demonstrating this balance with some toy code examples/experiments that I will create myself.

### Consideration 1: Similarity to Downstream Task

**[TENTATIVE MAIN IDEA, HASH OUT DETAILS AND ADD EXPERIMENTS/EXAMPLES LATER]**{: style="color: red; opacity: 0.80;" }: We want the scout tasks to be easier because this allows for better convergence of scouts, and convergence is needed if we want to make meaningful guide value calculations. Will include toy examples where scouting task is very different from target downstream tasks vs. where it's similar to target downstream tasks, and compare performances.


### Consideration 2: Ease of Convergence

**[TENTATIVE MAIN IDEA, HASH OUT DETAILS AND ADD EXPERIMENETS/EXAMPLES LATER]**{: style="color: red; opacity: 0.80;" }: We want the scout tasks to be similar because to the target downstream task, because the more similar the scout tasks are to the downstream task, the better the inductive biases will transfer over. So how do you make the scouting task easier? The two main ways are: 
- Make the training dataset for the scouts larger than for the downstream task. This is similar to traditional pre-training. 
- If your target task is a classification task, you can make the scout task have fewer classication categories to predict than the downstream task. 

Chossing the optimal downstream task is a balance between consideration 1 and 2. Will nclude toy examples where scouting task is very similar to the target downstream few-shot learning task but too difficult (almost as difficult as downstream task). Will show that this performs worse than GTL trained on easier task due to inability for scouts to converge.


## Is GTL Helpful in All Contexts?: An Exploration

**[TENTATIVE PLAN FOR THIS SECTION]**{: style="color: red; opacity: 0.80;" }: In the last section, I'll also be exploring how GTL can be adapted to and how it performs in various contexts that were NOT epxlored in the original GTL paper, with a focus on self-supervised contexts and complex architectures. I'll be including some experiemments I will create myself to demonstrate the effectiveness/ineffecitveness/nuances of GTL application in such contexts.

### Unsupervised Pre-training, Supervised Fine-tuning

**[TENTATIVE MAIN IDEA, HASH OUT DETAILS AND ADD EXPERIMENTS LATER]**{: style="color: red; opacity: 0.80;" }: The original GTL paper only demonstrated GTL that involved supervised scouting and supervised donwstream task. In many scenarios, again, especially in biomedicine, we don't have a large enough labeled dataset for pre-training either. Therefore, pre-training data must be unsupervised, but the downstream task will be supervised. This is challenging because the downstream task and scouting task should be similar and use the same/almost the same architecture so that guide values/inductive biases can trasnfer over comprehensively and effectively. I'll propose some of my ideas on how to deal with such scenarios, and whether or not GTL is as effective in this context compared to the examples demonstrated in the original paper. 

### Sophisticated Architectures With Built-in Inductive Biases

**[TENTATIVE MAIN IDEA, HASH OUT DETAILS AND ADD EXPERIMENTS LATER]**{: style="color: red; opacity: 0.80;" }: The original GTL paper only used small MLPs to demonstrate the effectiveness of MLP. I'm curious as to whether or not GTL will be as effective when applied to more sophistacted architectures that already have their own *built-in* inductive biases, e.g. CNNs and GNNs. I'll probably run some experimenets that are similar to the ones in the paper, but replacing MLPs with CNNs/GNNs. 
