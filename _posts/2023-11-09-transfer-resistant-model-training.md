---
layout: distill
title: Transfer Resistant Model Training
description: This blog post details our work on training neural networks that
  are resistant to transfer learning techniques.
date: 2023-12-12
htmlwidgets: true

# Anonymize when submitting
# authors:
#   - name: Anonymous

authors:
  - name: Ryan Yang
    url: "https://www.google.com/url?sa=i&url=https%3A%2F%2Fmanipulation.csail.mit.edu%2FFall2023%2Findex.html&psig=AOvVaw3MuJLCZwr7MxMiaaFQTBeC&ust=1699601771753000&source=images&cd=vfe&opi=89978449&ved=0CBIQjRxqFwoTCNil45C0toIDFQAAAAAdAAAAABAH"
    affiliations:
      name: MIT
  - name: Evan Seeyave
    url: ""
    affiliations:
      name: MIT

# must be the exact same name as your blogpost
bibliography: 2023-11-09-transfer-resistant-model-training.bib

# Add a table of contents to your post.
#   - make sure that TOC names match the actual section names
#     for hyperlinks within the post to work correctly.
toc:
  - name: Introduction and Motivation
  - name: Related Works
  - name: Methods
  - name: Experiments
  - name: Results
  - name: Discussion
  - name: Limitations and Future Work
  - name: Conclusion
  - name: Appendix
---

## Introduction and Motivation

In transfer learning, a model is trained for a specific task and is then fine-tuned for a different task <d-cite key="zhuang2020comprehensive"></d-cite>. In doing so, one tries to best leverage and reuse features and performance of the large pre-trained model for other tasks. Many works have focused on making transfer learning more robust and efficient. Transfer learning can be very useful for saving compute resources, time, and money.

In this project, we study an opposing question: how to learn model weights that classify well for one dataset but reduce learning efficiency when transferred to another. The motivation is as follows. As computational resources and capable models become more accessible, the risk of unregulated agents fine-tuning existing models increases, including for malicious tasks. Recent work has shown that previously aligned models can be compromised to produce malicious or harmful outputs <d-cite key="anonymous2023shadow"></d-cite> <d-cite key="qi2023finetuning"></d-cite>. This may even occur with a few adversarial examples against models specifically trained to produce safe outputs <d-cite key="lermen2023lora"></d-cite>. Currently, risks with language models are commonly discussed. However, investigating CNNs can guide designing defenses for neural network architectures against malicious agents in general.

{% include figure.html path="assets/img/2023-11-09-transfer-resistant-model-training/setting.png" class="img-fluid" %}

To our knowledge, there exists no previous literature on learning parameters robust against transfer learning. A related field is machine unlearning. In machine unlearning, a model must forget certain pieces of data used in training <d-cite key="cao2015towards"></d-cite> <d-cite key="10.1007/s42979-023-01767-4"></d-cite>. However, we wish to examine methods that not only guarantee poor performance after unlearning, but also after fine-tuning on the “malicious” or “forget” dataset. For example, using a popular unlearning approach which reaches 0% accuracy on the “forget” dataset, we easily fine-tuned the model with the same dataset to reach higher accuracy after a few epochs as shown below <d-cite key="tarun2023fast"></d-cite>. This is a gap in previous work in machine unlearning and demonstrates the novelty and difficulty of learning models that not only perform poorly on specified datasets but are robust against fine-tuning.

{% include figure.html path="assets/img/2023-11-09-transfer-resistant-model-training/machine_unlearning.png" class="img-fluid" %}

We propose two new approaches: selective knowledge distillation (SKD) and Reverse Model-Agnostic Meta-Learning (MAML) <d-cite key="finn2017model"></d-cite>. In SKD, a “student” model is trained using activations of a “teacher” for the beneficial data and trained on hardcoded activations for the “malicious” data. In Reverse-MAML, we attempt to learn parameters that aren’t robust to transfer to specified tasks. Due to computational constraints, we examine a toy setting with the CIFAR-10 Dataset as well as using a small CNN model shown in the appendix <d-cite key="krizhevsky2012imagenet"></d-cite>. Overall, both the Reverse-MAML and SKD approach exceed baseline approaches on scoring good accuracy on a “beneficial” dataset while being on-par with preventing fine-tuning on a “malicious” dataset. Thus, there remain limitations, and we conclude with future work.

## Related Works

### 1. Transfer Learning

As mentioned previously, transfer learning has been a long-time objective in deep learning research <d-cite key="zhuang2020comprehensive"></d-cite> <d-cite key="raffel2020exploring"></d-cite>. By training a model on one dataset, the goal is to be able to reuse parameters and learned features to achieve high performance or efficient learning for another dataset. Transfer learning for convolutional neural networks has been a popular approach, allowing users to train a high-performance model with limited computational resources or data <d-cite key="zhuang2020comprehensive"></d-cite>. Further work has analyzed settings for successful transfer learning in image classification and further challenges when there is poor transfer <d-cite key="plested2022deep"></d-cite>.

### 2. Model-Agnostic Meta-Learning (MAML)

MAML is an algorithm that makes models readily adaptable to new tasks <d-cite key="finn2017model"></d-cite>. It essentially primes the model for transfer learning as effectively as possible. The algorithm attempts to learn parameters and model weights such that a few steps of gradient descent on learning a new task will lead to good performance on said new task. Further work has continued attempting to meta-learn useful model parameters, building off of MAML <d-cite key="goerttler2021exploring"></d-cite> <d-cite key="park2019meta"></d-cite>.

### 3. Machine Unlearning

A closely aligned question to ours is the problem of machine unlearning. Machine unlearning attempts to remove the influence of a set of data points on an already trained model. In this setting, a model is initially trained on some dataset  <d-cite key="bourtoule2021machine"></d-cite>   <d-cite key="cao2015towards"></d-cite>      <d-cite key="10.1007/s42979-023-01767-4"></d-cite>. The model embeds information about and “remembers” features about data points. This means that the model uses information about the data points to make decisions. For example, LLMs like GPT can learn sensitive information about some people  <d-cite key="wu2023unveiling"></d-cite>. This might pose a threat to privacy. We may want the model to “forget” some subset of the training set, in this case information about the people. However, we currently have no standardized method of doing this. Machine unlearning is a nascent field in artificial intelligence research and is currently being studied. It is a difficult problem, and our work is tangential to machine unlearning.

<br />
<br />

To our knowledge, there hasn’t been any research on models that are resistant to transfer learning and fine-tuning. The works mentioned above, transfer learning techniques and MAML, focus on improving fine-tuning. We aim to make fine-tuning more difficult while preserving robustness on the original task. Machine unlearning seeks to forget data that the model has been previously trained on. On the other hand, our goal is to preemptively guard the model from learning certain data in the first place. Thus, our research question demonstrates a clear gap in existing research which has focused on either improving transfer learning or only reducing model performance on external datasets. Our research explores this new question in the deep learning field and draws from recent works to guide methodology.

## Methods

We propose three methods, one existing and two novel, to begin addressing the problem of learning parameters scoring high accuracy on a “beneficial” dataset but are robust against transfer learning on a known “malicious” dataset. Further experimental details are found in the experiments section.

### 1. Machine Unlearning

The first approach is a baseline and reimplementation of a popular machine unlearning method from <d-cite key="tarun2023fast"></d-cite>. Here, the model is initially trained on both the “beneficial” and “malicious” dataset and undergoes a forgetting stage where the “malicious” dataset is forgotten using a noise matrix. A final repair stage is then conducted to improve performance of the model on the “beneficial” dataset. Specific details can be found at <d-cite key="tarun2023fast"></d-cite>.

{% include figure.html path="assets/img/2023-11-09-transfer-resistant-model-training/performance.png" class="img-fluid" %}

### 2. Selective Knowledge Distillation

Our first proposed novel approach is selective knowledge distillation (SKD) drawing inspiration from knowledge distillation. In knowledge distillation, a smaller “student” model is trained to imitate a larger “teacher” model by learning logits outputs from the “teacher” model. In doing so, the “student” model can hopefully achieve similar performance to the “teacher” model while reducing model size and complexity.

{% include figure.html path="assets/img/2023-11-09-transfer-resistant-model-training/teacher_student.gif" class="img-fluid" %}

In SKD, we similarly have a “teacher” and “student” model. The “teacher” is a model that has high accuracy on the “beneficial” dataset but is not necessarily robust against fine-tuning on the “malicious” dataset. Our “student” model is almost identical in architecture to the “teacher” but excludes the final classification layer and the ReLU layer before it. This is shown below.

{% include figure.html path="assets/img/2023-11-09-transfer-resistant-model-training/teacher_student_architecture.png" class="img-fluid" %}

Our goal is for the student model to have high performance on the “beneficial” dataset after adding a classification layer while being robust against fine-tuning on the “malicious” dataset. To perform SKD, we initially train the teacher model until reaching sufficiently high performance on the “beneficial” dataset.

We then construct a dataset that contains all the images in the “beneficial” dataset. The labels are activations of the second-to-last layer of the “teacher” model. Note that this is similar to knowledge distillation, except we are taking the second-to-last layer’s activations. We further add all the images in the “malicious” dataset and set their labels to be a vector of significantly negative values. For our experiments, we used -100.0. We train the student model on this collective dataset of images and activation values.

{% include figure.html path="assets/img/2023-11-09-transfer-resistant-model-training/teacher_student_complex.gif" class="img-fluid" %}

Finally, we add a fully-connected classification layer to the student model and backpropagate only on the added layer with the “beneficial” dataset.

{% include figure.html path="assets/img/2023-11-09-transfer-resistant-model-training/student.png" class="img-fluid" %}

Our end goal is to prevent fine-tuning of our CNN on the “malicious” dataset. Thus, if the student model can output activations that all are negative if the image belongs in the “malicious” dataset, then after appending the ReLU layer and setting biases of the second-to-last layer to 0, the inputs to the final classification layer will always be 0, reducing the ability to learn on the “malicious” dataset. Furthermore, the gradient will always be 0 on inputs from the “malicious” dataset so any backpropagating on images and labels originating from the “malicious” dataset from the final layer activations would be useless.

### 3. Reverse-MAML

Recall that MAML is focused on finding some optimal set of model weights $$\theta$$ such that running gradient descent on the model from a new few-shot learning task results in a $$\theta’$$ that scores high accuracy on the new task <d-cite key="finn2017model"></d-cite>. MAML achieves this by learning the optimal $$\theta$$. To learn this $$\theta$$, MAML computes the second order gradient on the model weights. This allows the model to learn about where the initial $$\theta$$ should have been before an iteration of gradient descent so that taking the step of gradient descent would have led to the minimal loss.

{% include figure.html path="assets/img/2023-11-09-transfer-resistant-model-training/MAML.png" class="img-fluid" %}

In our version, we attempt to learn a $$\theta$$ that fine-tunes well to a data distribution $$p_1$$ but fine-tunes poorly to distribution $$p_2$$. To do this, we partition the data into two sets: a “good” set and a “bad” set. We train such that for “good” samples MAML performs the standard algorithm above, learning $$\theta$$ that would fine-tune well to the “good” samples. However, for the “bad” set we train the model to do the opposite, learning a $$\theta$$ that would lead to poor fine-tuning. To do this, when taking the second order gradient, the model goes up the gradient instead of down.

{% include figure.html path="assets/img/2023-11-09-transfer-resistant-model-training/reverse_MAML.png" class="img-fluid" %}

## Experiments

Due to computational constraints, we work in the following toy setting. We use the CIFAR-10 dataset where images in the first five ([0, 4]) classes are the “beneficial” dataset and the images in the last five ([5, 9]) classes are the “malicious” dataset. We split the 60,000 CIFAR-10 image dataset into a 40,000 image pre-training dataset, 10,000 image fine-tuning dataset, and 10,000 image test dataset. To evaluate each approach, we first evaluate the accuracy of the model on the beneficial test dataset. Then, we replace the last layer parameters of the output model, freeze all previous layer’s parameters, and finally fine-tune on the malicious fine-tuning dataset. We fine-tune using the Adam optimizer with a learning rate of 0.1 and momentum of 0.9. We finally evaluate model performance on a malicious test dataset. These steps in this evaluation represent the common pipeline to perform transfer learning and are shown below. Full hyperparameters for evaluation are listed in the appendix. We also perform ablation studies on the quality of the teacher model for SKD; further details are found in the Discussion section. All experiments, including ablations, are performed and averaged over 5 random seeds.

{% include figure.html path="assets/img/2023-11-09-transfer-resistant-model-training/pipeline.png" class="img-fluid" %}
{% include figure.html path="assets/img/2023-11-09-transfer-resistant-model-training/evaluation.png" class="img-fluid" %}

## Results

The first evaluation metric is accuracy of the outputted model from each approach on beneficial data. This is shown in the figure below.

{% include figure.html path="assets/img/2023-11-09-transfer-resistant-model-training/beneficial_accuracy.png" class="img-fluid" %}
<div class="caption">
   Figure 1 
</div>
The second metric of evaluation is the accuracy of the output model from each approach on test malicious data as it’s being fine-tuned on fine-tune malicious data. This is shown with learning curves in the figure below. Note that lower accuracy is better.

{% include figure.html path="assets/img/2023-11-09-transfer-resistant-model-training/malicious_accuracy.png" class="img-fluid" %}
<div class="caption">
   Figure 2 
</div>

## Discussion

We observe that finding parameters that have high accuracy on a “beneficial” dataset but are robust against fine-tuning on a “malicious” dataset is challenging. On all three methods, including a popular machine unlearning approach, the model is able to somewhat fit to the “malicious” dataset. However, for SKD, this accuracy consistently does not significantly exceed 40%.

More importantly, we find in Figure 1 that both Reverse-MAML and SKD are able score higher accuracy on the beneficial dataset. This is surprising as machine unlearning methods were designed to maintain high accuracy on a retain dataset. Combining these two graphs, we conclude that there remains future work to explain why the resulting models had such high accuracy on the malicious data out-of-the-box and how to minimize it.

We also experimented with Reverse-MAML under the Omniglot dataset <d-cite key="lake2015human"></d-cite>. Here, we attempted to fine-tune on digit images. We found that Reverse-MAML performed very well in this setting. After training the Reverse-MAML model, the model held around 85% test accuracy on the “Beneficial” Omniglot dataset and around 20% on the “Malicious” digit dataset. On the digit set, the model would often predict the same digit for all samples, as shown below. We believe that Reverse-MAML performed better here because the Omniglot characters and the digits are simpler to interpret and learn specific features about compared to CIFAR-10.

{% include figure.html path="assets/img/2023-11-09-transfer-resistant-model-training/digits.png" class="img-fluid" %}
<div class="caption">
    All digits were predicted to be a 2.
</div>

Slow learning in SKD is likely caused by filtering by the ReLU activation function which causes activations to become 0. This ideally occurs when we train the student model to output negative activation values into the final classification layer if the input is from the “malicious” dataset. These values make it more difficult to learn useful weights for the final classification layer and apply gradient descent on earlier layers. We confirm this by measuring misses or the percent of “malicious” images that don’t result in all 0 activations into the final classification layer shown below. We show, in general, misses are low across different teacher models. For this ablation, we vary teacher models by the number of epochs they are trained.

{% include figure.html path="assets/img/2023-11-09-transfer-resistant-model-training/student_table.png" class="img-fluid" %}

We also measure how accuracy of the teacher model impacts performance of the student downstream. We vary the number of epochs the teacher model is trained in and report accuracies of the teacher model on the “beneficial” dataset below. More importantly, we empirically show that high teacher accuracy on the “beneficial” dataset is needed for the student to achieve high accuracy on the “beneficial” dataset. This follows our knowledge distillation framework as the student attempts to mimic the teacher model’s performance on the “beneficial” dataset by learning activation values.

{% include figure.html path="assets/img/2023-11-09-transfer-resistant-model-training/error_bounds.png" class="img-fluid" %}

## Limitations and Future Work

### 1. Requirement for "Malicious" data

The motivating example for this project was preventing a malicious agent from hijacking a model to perform undesirable tasks. However, it is often not possible to list out every possible “bad” task, and thus future work which extends from this project can explore how to prevent fine-tuning of tasks that aren’t specified as clearly and completely.

### 2. Computational Restraints

Due to computational restraints, we were unable to test or fine-tune models with significantly higher parameter counts or experiment with larger datasets. However, this remains an important step as transfer learning or fine-tuning is commonly applied on large models which we could not sufficiently investigate. Thus, future work can apply these existing methods on larger models and datasets.

### 3. Exploration of More Methods in Machine Unlearning and Meta-Learning

Further analysis of existing methods in machine unlearning and meta-learning can be used to benchmark our proposed approaches. Though we tried to select methods that had significant impact and success in their respective problem settings, other approaches are promising, including using MAML variants like Reptile or FOMAML <d-cite key="DBLP:journals/corr/abs-1803-02999"></d-cite>.

### 4. Imperfection in filtering “malicious” data for SKD

Ideally, in SKD, the underlying model would always output negative activation values given a “malicious” input. However, this does not always occur, and thus fitting on the malicious data is still possible. Future work can explore how to improve this, though perfect accuracy will likely not be feasible. Furthermore, it is still possible for a malicious agent to hijack the model by performing distilled learning on the second-to-last layer activations, thus removing this ideal guarantee. Future work can also investigate how to have similar guarantees throughout all of the model’s activation layers instead of just one.

## Conclusion

In this project, we investigated how to train a model such that it performs well on a “beneficial” dataset but is robust against transfer learning on a “malicious” dataset. First, we show this is a challenging problem, as existing state of the art methods in machine unlearning are unable to prevent fine-tuning. We then propose two new approaches: Reverse-MAML and SKD. Both serve as a proof of concept with promising preliminary results on the CIFAR-10 Dataset. We conclude by noting there are limitations to this work, most notably the need for a “malicious” dataset and computational limits.  We then propose future work stemming from these experiments.

## Appendix


CNN Architectures used for experiments:
{% include figure.html path="assets/img/2023-11-09-transfer-resistant-model-training/CNN_architectures.png" class="img-fluid" %}

* Note, all graphs and tables are averaged over 5 seeds with reported standard deviation.