---
layout: distill
title: Data Poisoning is Hitting a Wall
description: In this post, we look at the paper 'Data Poisoning Won't Save You From Facial Recognition', discuss the impact of the work, and additionally look at how this work fares in the current state of adversarial machine learning. Being a blog post as opposed to a traditional paper, we try to avoid inundating the reader with mathematical equations and complex terminologies. Instead, we aim to put forth this work's primary concept and implications, along with our observations, in a clear, concise manner. Don't want to go through the entire post? Check out the TL;DR at the end for a quick summary.

date: 2022-12-01
htmlwidgets: true

# Anonymize when submitting
authors:
  - name: Anonymous

bibliography: 2022-12-01-facial-poisoning.bib

toc:
  - name: Overview and Motivation
    subsections:
    - name: What is Data Poisoning?
  - name: Why doesn't Data Poisoning work?
  - name: High Level Idea
  - name: Experiments
    subsections:
    - name: Adaptive defenses break facial poisoning attacks
    - name: Attack Detection
    - name: Time is all you need
    - name: Robustness shouldnt come at the cost of accuracy
  - name: Conclusion
  - name: Outlook
  - name: TLDR

---

## Overview and Motivation

To illustrate the data poisoning process, and to tie in with the paper below, let's describe data poisoning against the backdrop of the facial recognition problem.

Facial recognition systems have been known to pose a severe threat to society. With unprecedented advancements in AI research, it is evident that this threat will be around for a while. There has been a steady increase in vendors offering facial recognition services for downstream applications — ranging from customer onboarding tools to criminal identification for police forces. The systems provided by these vendors are usually trained on images of users' faces scraped from the Web. Ethical and moral concerns aside, this poses a considerable risk to the privacy of individuals.

#### What is Data Poisoning?

Keeping this in mind, a growing body of work has emerged that allows users to fight back using principles from adversarial machine learning. Primary among these is the technique of data poisoning - where users can perturb pictures that they post online so that models that train on these become _poisoned_. In other words, once a model has been introduced to a perturbed image of a user, it misidentifies any future instances of that person.

Services like _Fawkes_ popularized this approach by offering a service promising "strong protection against unauthorized {facial recognition} models." Users could pass their images through Fawkes and receive poisoned photos - virtually identical to the naked eye, which were then posted to social media, alleviating any worries that they might be used to identify them in the future. It quickly gained popularity, was covered by the New York Times <d-footnote>[This tool could protect your photos from Facial Recognition](https://www.nytimes.com/2020/08/03/technology/fawkes-tool-protects-photos-from-facial-recognition.html)</d-footnote> and received over 500,000 downloads. Following Fawkes' success, similar systems were proposed in academic and commercial settings.

{% include figure.html path="assets/img/2022-12-01-facial-poisoning/facial_poisoning.png" class="img-fluid" %}

*** 
The authors of the paper, however, look at these systems from a different perspective. They argue that services like Fawkes (and poisoning strategies in general) cannot protect users' privacy when it comes to facial recognition systems. In fact, it usually exacerbates the situation by providing them with a false sense of security.
For instance, there might have previously been a privacy-focused user who would have refrained from uploading their photos to the Internet. However, they might do so now under the false belief that their poisoned photos would work towards protecting their privacy. Thus, these users are now _less private_ than they were before.

## Why doesn't data poisoning work?

While data poisoning may have uses in other fields, such as healthcare, this post shows that it would not protect against facial recognition models. The main reason for this is due to a fundamental asymmetry between the users and the model trainers. Let us take the scenario described in the above figure. A user commits to an attack and uploads a perturbed image of themselves to the Web. This image eventually gets scraped by the model as part of its data collection strategy. In this case, the model trainer, or the vendors offering facial recognition services, now benefit from acting second. This provides them with two significant advantages:
-   Since image poisoning systems cater to large user bases, these systems are usually made publicly accessible. This allows the model trainers to become aware of the technique used, which, in turn, helps them apply techniques to resist the poisoning attacks. This strategy of using alternate training techniques is known as an **adaptive defense**.

-   As current poisoning attacks are designed to prevent _existing_ facial recognition tools from working, there is no reason to assume that future models will also be poisoned. So, trainers can simply wait a while and use newer models to keep identifying users, which would be invulnerable to poisoning attacks. This technique can (aptly) be named an **oblivious defense**.

Observant readers might equate this setting of continually evolving attack and defense tactics to an _arms race_. However, since a perturbation applied to an image cannot be changed once scraped by the model, a successful attack has to remain effective against _all_ future models, even those trained adaptively against the attack. A better alternative to this would be pushing for legislation that restricts the use of privacy-invasive facial recognition systems.

## High Level Idea
We now look at the conclusions put forward in the excellent paper written by Radiya-Dixit _et al_. 

 1. An adaptive model trainer with black-box access to facial recognition systems like Fawkes can train a robust model that resists poisoning attacks and correctly identifies all users with high accuracy.
 2. An adaptive model trainer can also repurpose this model to _detect_ perturbed pictures with near-perfect accuracy.
 3. Image poisoning systems have already been broken by newer facial recognition that appeared less than a year after the attacks were introduced and employed superior training strategies.
4.  It is possible to increase the robustness of a model (against poisoning attacks) without degrading its accuracy in identifying 'clean' images.

Let us take a closer look and deconstruct how the authors arrived at these conclusions.

## Experiments 

For clarity, before we arrive at the individual conclusions, we look at the setup used by the authors to carry out their experiments.

The authors evaluate three distinct poisoning attacks: **Fawkes v0.3**, **Fawkes v1.0**<d-cite key="shan2020fawkes"></d-cite>, and a separate attack published at ICLR 2021 called **LowKey**<d-cite key="cherepanova2021lowkey"></d-cite>. All of these function on the same underlying principle of data poisoning. Their goal is to force the facial recognition model to associate an image with spurious features absent in unperturbed images.

The experiments are performed with the _FaceScrub_ dataset<d-cite key="ng2014data"></d-cite>, which contains over 50,000 pictures of 530 celebrities. A sample run of an experimental procedure can be described as follows:
A user, in this case, one of the celebrities in the _FaceScrub_ dataset, perturbs all of their images with _Fawkes_ or _LowKey_ in their strongest settings. These images then end up as the training data used by the model trainer. The model trainer uses the standard approach for training their facial recognition system by employing a pre-trained feature extractor to convert pictures into embeddings. Given a test image, the model tries to find a training example that minimizes the distance between them in the embedding space and returns the identity associated with the training example.

The authors use various models as feature extractors from _FaceNet_<d-cite key="schroff2015facenet"></d-cite> to OpenAI's _CLIP_<d-cite key="radford2021learning"></d-cite>. This is an important step that helps quantify the effectiveness of the **oblivious defense** strategy.
***

#### Adaptive defenses break facial poisoning attacks

This section describes how the model trainer can adaptively train a generic feature extractor that can resist poisoning attacks.

The model trainer begins by collecting a public dataset of unperturbed images. In this case, that would be a canonical dataset of celebrities that are a part of the _FaceScrub_ dataset. With black-box access to the poisoning tool, the trainer calls it to obtain perturbed samples of the same images.

{% include figure.html path="assets/img/2022-12-01-facial-poisoning/adaptive-attack.gif" class="img-fluid" %}

With access to both unperturbed images and their corresponding poisoned counterparts, the trainer can teach a model to produce similar embeddings for both sets of pictures, encouraging the model to adaptively learn robust features. This is done hoping that this robustness would eventually generalize to perturbations' applied to other images.

<blockquote>
While the above strategy works in theory, it requires direct intervention from model trainers by using the 'clean' images provided by them. This would not scale well, especially for large-scale facial recognition systems that look at millions of photographs. However, this attack could also occur without the trainers' explicit involvement. There is a high possibility that some users already have unperturbed images of themselves on the Web; either they forgot to perturb some pictures, or they were uploaded by someone else. Feature extractors trained on these pictures would then be encouraged to learn robust features.
</blockquote>

**Results:** All three attacks were evaluated against a non-robust _WebFace_ model to establish a baseline. They were found to have a misclassification rate of 55-77% for users who poisoned their pictures online. This compares starkly to a rate of 8% for unprotected users. However, when trained adaptively, the misclassification rate for all users - protected or unprotected - dropped to 5-8%, and all poisoning attacks were rendered ineffective.
***

#### Attack Detection

Since the model trainers have black-box access to the facial poisoning tools (_Fawkes_ and _LowKey_), they can also turn the tables and build a detector to determine whether a specific image has been perturbed. Such a detector can dynamically filter out perturbed photos, allowing the model to retain only unperturbed pictures of a user. Moreover, detecting an attack could be a privacy concern (for instance, law enforcement might actively target users whose attack attempts are detected).

To verify this, the authors were able to fine-tune a standard pre-trained _ImageNet_ model to distinguish between perturbed and clean images of 25 random celebrities in the dataset. The model detected the poisoned images with near-perfect precision (99.8%) and recall (99.8%).
***

#### Time is all you need

Rather than creating poisoned counterparts to clean images and adaptively training a model, trainers have a much simpler alternative. They can simply wait for better facial recognition systems to be developed and then retroactively apply such a system to pictures they scraped in the past. _**Simply put, facial poisoning attacks cannot withstand the test of time**_.

To bypass this _oblivious_ defense strategy, an attack must not only be able to fool all present models but also be effective against future iterations without changing its perturbation. Asymetrically (to the benefit of the model trainer) newer techniques need not be robust to all attacks; instead, they just have to resist the specific method used in previous pictures.

{% include figure.html path="assets/img/2022-12-01-facial-poisoning/oblivious-attack.gif" class="img-fluid" %}

To confirm this, the paper included a study where _Fawkes_ was pitted against various feature extractors ordered chronologically. While the original _Fawkes v0.3_ was utterly ineffective against any model apart from _WebFace_, the updated v1.0 could transfer its attack to other extractors like _VGGFace_, _FaceNet_, and _ArcFace_. However, while _Fawkes v1.0_ provided a perfect (100%) error rate on the _Celeb1M_ model (the one it was trained to target), it failed miserably against more recent extractors like _MagFace_<d-cite key="meng2021magface"></d-cite> or _CLIP_. A similar trend was also observed when using _LowKey_. While it fared better than _Fawkes_ and could transfer its attack to MagSafe, LowKey failed to break the fine-tuned _CLIP_ model trained by the authors.

To provide more credence to their findings, the authors also illustrated how users who downloaded an older model (_Fawkes v0.3_, for example) could not 'regain' their privacy by switching to an updated attack. For brevity, this post does not go into the specifics, but we encourage interested readers to look at the paper and additional supplementary material.
***

#### Robustness shouldn't come at the cost of accuracy

A potential caveat for the _adaptive_ and _oblivious_ defenses is that increased robustness may come at the cost of decreased accuracy. For example, the CLIP model is much more robust than all the other feature extractors, but its clean accuracy falls slightly below the best models. In most cases, a trainer might be hesitant to deploy a _CLIP_-based model if only a small minority of users try to attack the system.

Keeping this in mind, the authors demonstrated two approaches that allow model trainers to incorporate the best qualities of both worlds:

**Top2:** This approach involved having a human in the loop. The authors propose that the system simply run the image through both models and return two candidate labels. To further streamline the process, the system could pass the image to the robust model only when the more accurate model cannot get a result. Humans could visually inspect these images to check for inconsistencies or determine if they were poisoned.

**Confidence Thresholding:** To automate the above process, the system could begin by passing the image through the most accurate model and checking the prediction's confidence. This can be quantitatively defined as the distance between the target picture and its nearest neighbor in the embedding space. If the system finds the confidence below a certain threshold, the image is passed through the robust model instead.

The paper demonstrates a facial recognition system that uses _MagFace_ for an accurate model and combines that with a more robust model like the fine-tuned _CLIP_ or an adaptively trained model. In both cases, the clean accuracy of the system matches or exceeds that of _MagFace_, while retaining high robustness to attacks.

***

### Conclusion

The main takeaway from this post is that data poisoning is no longer an effective method to protect users from facial recognition systems. The original premise for developing poisoning attacks was to facilitate an 'arms race,' where better attacks could counteract improved defenses. However, the people who deploy facial recognition models would always have the upper hand. 

The paper shows that facial recognition models can be trained to detect and overcome poisoning attacks by simply having black-box access to a public-facing tool or just waiting for newer models and retroactively using them. To compete even against the latter category of systems, users would have to presume that minimal changes will be made to facial recognition models in the upcoming years. Given the state and pace of research in the field, that seems highly unlikely. 
***

### Outlook
This blog post provides a better understanding of the techniques used to neutralize the effects of data poisoning from the ICLR 2022 paper _Data Poisoning Won't Save You from Facial Recognition._ We hope that this has been of help to researchers and practitioners in the fields of adversarial ML.

We now look to provide some clarifications and how we think this work would fit in the current age of machine learning.

**The work is a net positive** 
This paper takes a gloomy stance on the current state of protection against facial recognition models. By stating that model trainers would always have the upper hand in the race by simply switching to a more advanced framework, the authors quash any possibility of a technological solution. Instead, they argue that a legislative approach might hold the key to solving the problem. Looking at the discussion between the authors and the reviewers before the acceptance of the paper <d-footnote>[ICLR OpenReview](https://openreview.net/forum?id=B5XahNLmna)</d-footnote>, it was clear that the reviewers were reluctant to accept the finality of the solution - a sentiment we're sure would be shared by many others. However, if nothing else, this paper warns users against the futility of using commercial products like Fawkes to protect their identities. In alleviating the false sense of security provided by data poisoning attacks, this paper - and, by extension, this post - serves as a net positive for users' privacy. 

**Is legislation the answer?**
With artificial intelligence embedding itself into society at an unprecedented rate, it is clear that a complete overhaul of legislative frameworks is urgently required. As AI becomes more mainstream, privacy-invasive systems could graduate from storing information to using them for financial incentives. While we have seen this happen with users' browsing data, the repercussions of using biometrics would be much more severe. In fact, there have already been cases where facial recognition has been used by companies on users without their prior explicit consent. <d-footnote> [Madison Square Garden has put lawyers who represent people suing it on an 'exclusion list' to keep them out of concerts and sporting events](https://www.nytimes.com/2022/12/22/nyregion/madison-square-garden-facial-recognition.html)</d-footnote>

While we agree with the authors for a push towards proper legislation, given the rate of progress, we believe the community can do more. Legislation is a process that moves slowly and usually needs uniform implementation. Literature on the subject has shown that each country has its own views on the emerging landscape of AI <d-footnote>[How different countries view artificial intelligence](https://www.brookings.edu/research/how-different-countries-view-artificial-intelligence/)</d-footnote> and bases its rules on those views. These may or may not always work. We believe a temporary stopgap in the form of a technological solution would be helpful, while a legislative solution holds maximum promise in the long run.

***


### TL;DR


 This post broadly explores the ineffectiveness of data poisoning strategies against facial recognition models. It shows that commercial solutions like Fawkes and LowKey, which allow users to perturb their photos before posting them to social media, offer no protection to the users once their pictures are scraped.
 
It reveals that an 'oblivious' model trainer can simply wait long enough for future developments to nullify the effects of the perturbation. Or, since the people developing the facial recognition systems also have access to poisoning tools, they can simply develop strategies to detect and adapt to the perturbations. 

Finally, given that there are no technical solutions to the problem, the best approach would be to push for legislation to counteract privacy-invasive facial recognition systems.

***
