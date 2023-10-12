---
layout: distill
title: 'Underfitting and Regularization: Finding the Right Balance'
description: In this blog post, we will go over the ICLR 2022 paper titled NETWORK AUGMENTATION FOR TINY DEEP LEARNING. This paper introduces a new training method for improving the performance of tiny neural networks. NetAug augments the network (reverse dropout), it puts the tiny model into larger models and encourages it to work as a sub-model of larger models to get extra supervision.
date: 2023-01-15
htmlwidgets: true

# Anonymize when submitting
authors:
  - name: Anonymous


# must be the exact same name as your blogpost
bibliography: 2023-01-15-Underfitting-and-Regularization-Finding-the-Right-Balance.bib

# Add a table of contents to your post.
#   - make sure that TOC names match the actual section names
#     for hyperlinks within the post to work correctly.
toc:
  - name: Goal 
  - name: Background
    # subsections:
    #   - name: Interactive Figures
  - name: Pitfalls
  - name: Formulation of NetAug
  - name: Introducing NetAug with Relation Knowledge Distribution (RKD)
    subsections:
      - name: Loss Functions in RKD
      - name: Combining RKD with NetAug
      - name: NetAug Training with RKD
  - name: Fasten Auxillary Model Training
  - name: Generating largest augmented model via Single Path One-Shot NAS
  - name: Evaluation and Inference
  - name: Conclusion

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
  .small_img {
    width: 50%;
    height: auto;
    margin-left: auto;
    margin-right: auto;
  }
---

## Goal of this blog post

Network Augmentation aka NetAug <d-cite key="DBLP:conf/iclr/CaiG0022"></d-cite> caters to training small neural network architectures like MobileNetV2-Tiny for the best top-k percent accuracy. The paper argues that training small neural networks technically differs from that of large neural networks because the former is prone to underfitting. NetAug is contrary to traditional methods like dropout<d-cite key="10.5555/2627435.2670313"></d-cite>, network pruning<d-cite key="9043731"></d-cite>, quantization<d-cite key="jacob2018quantization"></d-cite>, data augmentation<d-cite key="https://doi.org/10.48550/arxiv.1712.04621"></d-cite> and other regularization techniques<d-cite key="10.1007/s10462-019-09784-7"></d-cite> NetAug can be viewed as reversed form of dropout, as we enlarge the target model during the training phase instead of shrinking it. In this blog post, we identify some pitfalls with NetAug and propose potential workarounds.

## Background

NetAug solely focuses on improving the performance of the tiny neural networks during inference, whilst optimizing their memory footprint to deploy them on edge devices. Tiny neural networks are usually inclined to underfit. Hence, the traditional training paradigms will not work for these small models because they fundamentally tackle the problem of overfitting and not overcome the underfitting issue. Several techniques in the recent time like data augmentation, pruning, dropout, knowledge distillation have been proposed to improve the generalizability of neural networks.

1. **Knowledge Distillation**<d-cite key="hinton2015distilling"></d-cite>
   : It is quite difficult to deploy and maintain an ensemble. However, previous research has shown that it is possible to learn a single cumbersome model that has the same performance as an ensemble. In most knowledge distillation methods there exist a large teacher model that transfers its knowledge as a learned mapping via training to a small student model with the teacher models output. There exists several techniques like self-distillation in which the teacher model trains itself continuously. Convectional KD methods try to optimise the objective function such that the loss function penalizes the difference between the student and teacher model.

2. **Regularization**<d-cite key="10.1007/s10462-019-09784-7"></d-cite>
   : Regularization is used to prevent overfitting of any ML model by reducing the variance, penalizing the model coefficients and complexity of the model. Regularization techniques are mainly composed of data augmentation one of the most simplest and conveninet ways to expand the size of the dataset such that it prevents overfitting issues that occur with a relatively small dataset. Dropout is also popularly applied while training models, in which at every iteration incoming and outgoing connections between certain nodes are randomly dropped based on a particular probability and the remaining neural network is trained normally.

3. **Tiny Deep learning**<d-cite key="lin2020mcunet"></d-cite>,<d-cite key="lin2021mcunetv2"></d-cite>,<d-cite key="lin2022ondevice"></d-cite>
   : Several challenges are paved while transitioning from conventional high end ML systems to low level clients, maintaining the accuracy of learning models, provide train-to-deploy facility in resource economical tiny edge devices, optimizing processing capacity. This method includes AutoML procedures for designing automatic techniques for architecturing apt neural netowrks for a given target hardware platform includes customised fast trained models,auto channel pruning methods and auto mixed precision quantization. The other approaches like AutoAugment methods automatically searches for improvised data augmentation within the network to prevent overfitting. There exists network slimming methods to reduce model size, decrease the run-time memory footprint and computing resource.

4. **Network Augmentation**<d-cite key="DBLP:conf/iclr/CaiG0022"></d-cite>
   : This method was proposed to solve the problem of underfitting in tiny neural networks. This is done by augmenting the given model (referred to as base model) into a larger model and encourage it to work as a sub-model to get extra supervision in additoin to functioning independently. This will help in increasing the representation power of the base model because of the gradient flow from the larger model and it can be viewed equivalently as "**reverse-dropout**".

5. **Neural Architechture Search (NAS)**<d-cite key="https://doi.org/10.48550/arxiv.1808.05377"></d-cite>
   : This method was proposed to solve the problem of architecture optimization and weight optimization based on the underlying training data. NAS usually involves defining an architectural search space and then searching for the best architecture based on the performance on the validation
   set. Recent approaches include weight sharing, nested optimization and joint optimization during training. However, there are drawbacks in using these approaches because they are computationally expensive and suffer from coupling between architecture parameters and model weights. This will degrade the performance of the inherited weights.

## Formulation of NetAug

The end goal of any ML model is to be able to minimize the loss function with the help of gradient descent. Since tiny neural network have a very small capacity the gradient descent is likely to get stuck in local minima. Instead of traditional regularization techniques which add noise to data and model, NetAug proposes a way to increase the capacity of the tiny model without changing its architecture for efficient deployment and inference on edge devices. This is done by augmenting the tiny model (referred to as base model) into a larger model and jointly training both the base model independently and also the augmented model so that the base model benefits from the extra supervision it receives from the augmented model. However, during inference only the base model is used.

To speed-up the training, a single largest augmented model is constructed by augmenting the width of the each layer of the base model using an augmentation factor $$r$$. After building the largest augmented model, we construct other augmented models by selecting a subset of channels from the largest augmented model. NetAug proposes a hyper-parameter $$s$$, named diversity factor, to control the number of augmented model configurations. We set the augmented widths to be linearly spaced between $$w$$ and $$r \times w$$. For instance, with $$r = 3$$ and $$s = 2$$, the possible widths would be $$[w, 2w, 3w]$$.

## Pitfalls in NetAug

1. **Training the Generated Augmented Models**{:#training_aug_models}
: NetAug randomly samples sub-models from the largest model by augmenting the width instead of depth, its highly important to ensure we speed up the training time and reduce the number of traning iterations for these generated sub-models thereby enhancing the training convergence and reaching optimization faster. We theoretically aim at aiding this by introducing a re-parametrisation technique during training that involves sharing and unsharing of weights to attain convergence much faster.
<!--
Rough idea for solution: The parameter s is the diversity factor, to control the number of augmented model configurations. Try to penalise/introduce a reward for s to favor a particular configurations that perform better/provide better auxiliary supervision.
 -->
2. **Naive Loss Function**{:#naive_loss}
: NetAug computes loss in a very trivial form, i.e, by simply performing a weighted sum over the loss of the base model with that from the respective sampled augmented models. However, it was mentioned in the paper that sampling more than one sub-models from the largest augmented model in each training step is resulting in degradation of the base model's performance. This can be attributed to the fact that simple weighted sum of losses from the base supervision and auxiliary supervision is causing the auxiliary supervision to shadow the base model. We propose different mixing strategies to circumvent this problem.
<!--3. **Overfitting of the Augmented Models**: As the network grows lineary even the number of augmented models increase, thereby overfitting during the training process. Having large number of such models with a poor representation among them will simply increase the training overhead and will cause haphazard confusion while selecting the largest augmented model. NetAug focuses on design and implementing a single best largest, going ahead with just the largest paramater model would yield in bad accuracy and will worsen the overall model performance.-->
3. **Generating the Largest Augmented Model**{:#generating_aug_model}
   : For a particular network, rather than tuning for just a single network hyperparameter (i.e., network, depth, width etc.), what if we instead tune all the closely relevant network hyperparameters for every augmented sub-model? To advocate this it's sensible to compare the entire distribution of hyperparameter across the model. This can be tackled using NAS to find the best largest augmented model and then use it for auxiliary supervision of the base model.

## Introducing NetAug with Relation Knowledge Distribution (RKD)

Knowledge distillation in learned models is constituted of:

1.  **Individual Knowledge Distillation**
    : Outputs of individual samples represented by the teacher and student are matched.
2.  **Relational Knowledge Distillation**
    : Relation among examples represented by the teacher and student are matched. RKD is a generalization of convectional knowledge distillation that combines with NetAug to boost the performance due to its complementarity with conventional KD, that aims at transferring structural knowledge using mutual relations of data examples in the teacher’s output presentation rather than individual output themselves. Contrary to conventional approaches called as Individual KD (IKD) that transfers individual outputs of the teacher model $$f_T(\cdot)$$ to the student model $$f_S(\cdot)$$ point-wise, RKD transfers relations of the outputs structure-wise and computes a relational potential $$\psi$$ for every $$n$$-tuple of data instance and transfers the relevant information through the potential from the teacher to the student models. In addition to knowledge represented in the output layers and the intermediate layers of a neural network, knowledge that captures the relationship between feature maps are also used to train a student model.

We specifically aim at training the teacher and student model in a online setting, in this online type of distillation training method both the teacher and the student model are trained together simultaneously.

{% include figure.html 
path="assets/img/2023-01-15-Underfitting-and-Regularization-Finding-the-Right-Balance/rkd.png" class="img-fluid" %}

**Relational Knowledge distillation can be expressed as** -  
 Given a teacher model $$\:T\:$$ and a student model $$S$$, we denote $$f_T(\cdot)$$ and $$f_S(\cdot)$$ as the functions of the teacher and the student, respectively, and $$\psi$$ as a function extracting the relation, we have

$$
\begin{equation}
\mathcal{L}_{\text{RKD}} = \sum \limits_{\{x_1, \ldots, x_n\} \in \chi^N} l \big(\psi(t_1,\cdots,t_n), \: \psi(s_1, \cdots, s_n)\big)
\end{equation}
$$

where $$\mathcal{L}_{\text{RKD}}$$ is the loss function, $$t_i = f_T(x_i)$$ and $$s_i = f_S(x_i)$$ and $$x_i \in \chi$$ denotes the input data.

### Loss Functions in RKD

1. **Distance-wise distillation loss (pair)**

<div class="small_img row mt-1">
{% include figure.html path="assets/img/2023-01-15-Underfitting-and-Regularization-Finding-the-Right-Balance/RKD-D.png" class="img-fluid" %}  
</div>

This method is known as RKD-D. It transfers relative distance between points on embedding space. Mathematically,
$$\begin{equation}\psi_{D}(t_i, t_j) = \frac{1}{\mu}\big\| t_i - t_j\big\|_2\end{equation}$$
where $$\psi_d(\cdot, \cdot)$$ denotes distance wise potential function
$$\begin{equation}\mu = \frac{1}{|\chi^2|}\sum\limits_{(x_i, x_j) \in \chi^2} \big\| t_i - t_j\big\|_2\end{equation}$$
$$\begin{equation}\boxed{\mathcal{L}_{\text{RKD-D}} = \sum \limits_{(x_i, x_j) \in \chi^2} l_\delta \big(\psi_D(t_i, t_j), \psi_D(s_i, s_j)\big)}\end{equation}$$
where $$l_{\delta}$$ denotes the Huber Los

$$
\begin{equation}l_\delta(x, y) = \begin{cases}
                              \frac{1}{2} (x-y)^2\:\:\:\: \text{for } |x-y| \leq 1 \\
                              |x - y| - \frac{1}{2}\:\:\: \text{otherwise.}
                              \end{cases}\end{equation}
$$

  2. **Angle-wise distillation loss (triplet)**

  <div class="small_img row mt-1">
      {% include figure.html path="assets/img/2023-01-15-Underfitting-and-Regularization-Finding-the-Right-Balance/RKD-A.png" class="img-fluid" %}  
  </div>

This method is known as RKD-A. RKD-A transfers angle formed by three points on embedding space. Mathematically,
$$\begin{equation}\psi_{A}(t_i, t_j, t_k) = \cos \angle t_it_jt_k = \langle \boldsymbol{e}^{ij}, \boldsymbol{e}{jk}\rangle\end{equation}$$
where $$\psi_A(\cdot, \cdot, \cdot)$$ denotes angle wise potential function
$$\begin{equation}\boldsymbol{e}^{ij} = \frac{t_i - t_j}{\big\|t_i - t_j\big\|_2}, \: \boldsymbol{e}^{jk} = \frac{t_k - t_j}{\big\|t_k - t_j\big\|_2}\end{equation}$$

$$
\begin{equation}\boxed{\mathcal{L}_{\text{RKD-A}} = \sum\limits_{(x_i, x_j, x_k) \in \chi^3} l_\delta \big(\psi_A(t_i, t_j, t_k), \psi_A(s_i, s_j, s_k)\big)}\end{equation}
$$

### Combining RKD with NetAug

We propose the following loss function to solve [naive loss problem of NetAug](#naive_loss)

$$\begin{equation}\mathcal{L}_{\text{aug}} = \underbrace{\mathcal{L}(W_t)}_{\text{base supervision}} \:+\: \underbrace{\alpha_1 \mathcal{L}([W_t, W_1]) + \cdots + \alpha_i \mathcal{L}([W_t, W_i]) + \cdots}_{\text{auxiliary supervision, working as a sub-model of augmented models}}  \:+\: \lambda_{\text{KD}}\,\underbrace{\mathbf{\mathcal{L}_{\text{RKD}}}}_{\text{relational knowledge distillation}}\label{eqn:loss_func}\end{equation}$$

where $$[W_t, W_i]$$ represents an augmented model where $$[W_t]$$ represents the tiny neural network and $$[W_i]$$ contains weight of the sub-model sampled from the largest augmented model, $$\alpha$$ is scaling hyper-parameter for combining loss from different augmented models and finally $$\lambda_{\text{KD}}$$ is a tunable hyperparameter to balance RKD and NetAug.

### NetAug Training with RKD

In NetAug, they train only one model for every epoch, training all the augmented models all once is not only computationally expensive but also impacts the performance. The proportion of the base supervision will decrease when we sample more augmented networks, which will make the training process biased toward augmented networks and shadows the base model.

To further enhance the auxiliary supervision, we propose to use RKD in an online setting i.e., the largest augmented model will act as a teacher and the base model will act as a student. Both the teacher and the student are trained simultaneously.

We train the both the base model and the augmented model via gradient descent based on the loss function $$\eqref{eqn:loss_func}$$. The gradient update for the base model is then given by

$$\begin{equation}W^{n+1}_t = W^n_t - \eta \bigg(\frac{\partial \mathcal{L}(W^n_t)}{\partial W^n_t} + \alpha \frac{\partial \mathcal{L}([W^n_t, W^n_i])}{\partial W^n_t} + \lambda \frac{\partial \mathcal{L}_{\text{RKD}}([W^n_t, W^n_l])}{\partial W^n_t}\bigg)\end{equation}$$

Similar update equations can be obtained for the largest augmented model and the sub-models as well.

## Fasten Auxillary Model Training

We propose this method to solve the problem [training the generated augmented models](#training_aug_models).
Based on <d-cite key="yang2021speeding"></d-cite> we propose to speed up the training process for the augmented sub models such that it can attain faster convergence and reduce the number of training iterations thereby obtain a better performance. In this technique, in the early phase of training, the neural network is trained with weights shared across all the layers of the model, to learn the commonly shared component across weights of different layers, and towards the later phase of training we un-share weights and continue training until convergence. Weight sharing for initial training steps will contrain the model complexity effectively. It brings the weights closer to the optimal value, which provides a better initialization for subsequent training steps and improved model generalization.

**Mathematical Formulation**

Denote the neural model as consisting of $$L$$ stacked structurally similar modules as $$\mathcal{M} = \{\mathcal{M}_i\}, \, i=1,\cdots, L$$ and $$w = \{w_i\}, \, i=1,\cdots, L$$ denote the corresponding weights. These weights are re-parametrized as

$$\begin{equation}w_i = \frac{1}{\sqrt{L}}w_0 + \tilde{w}_i, \:\:\:\: i=1,\cdots,L\end{equation}$$

Here $$w_0$$ represents the shared weights across all modules and is referred to as **stem-direction** and $$\tilde{w}_i$$ represents the unshared weights across all modules and is referred to as **branch-directions**.

**Training Strategy**

Denote $$T$$ as the number of training steps, $$\eta$$ as the step size and $$\alpha \in (0, 1)$$ is a tunable hyper-paramter indicating the fraction of weight sharing steps. Then we train $$\mathcal{M}$$ as follows:

- **Sharing weights in early stage:** For the first $$\tau = \alpha \cdot T$$ steps, we update the shared weights $$w_0$$ alone with gradient $$g_0$$
- **Unsharing weights in later stage:** For the next $$t \geq \alpha \cdot T$$, we update only the unshared weights $$\tilde{w}_i$$ with gradient $$\tilde{g}_i$$

The effective gradient updates for $$w_i$$ can be found using chain rule as follows:

$$
\begin{equation}g_0 = \frac{\partial \mathcal{L}}{\partial w_0} = \sum\limits_{i=1}^L \frac{\partial \mathcal{L}}{w_i}\,\frac{\partial w_i}{\partial w_0} = \frac{1}{\sqrt{L}}\sum\limits_{i=1}^L g_i
\end{equation}
$$

$$\begin{equation}\tilde{g}_i = \frac{\partial \mathcal{L}}{\partial \tilde{w}_i} = \frac{\partial \mathcal{L}}{\partial w_i}\,\frac{\partial w_i}{\partial \tilde{w}_i} = g_i\end{equation}$$

where $$g_i$$ denotes the gradients of $$w_i$$ and $$\mathcal{L}$$ denotes the loss function.

{% include figure.html path="assets/img/2023-01-15-Underfitting-and-Regularization-Finding-the-Right-Balance/NetAug_SWE.png" class="img-fluid" %}

## Generating largest augmented model via Single Path One-Shot NAS

We propose this method to solve the problem of [generating the largest augmented model](#generating_aug_model). In NetAug, the largest augment model is generated randomly just based on the hyperparameters $$r$$ and $$s$$. Single Path One-Shot NAS with uniform sampling <d-cite key="guo2020single"></d-cite> revists the pitfalls of weight coupling in previous weight sharing methods. The one-shot paradigm is made attractive for real world tasks and better generalization. It is hyperparameter free, single path strategy works well because it can decouple the weights of different operations. This implementation can be more efficient in multiple search space. Using this technique we generate largest optimized augmented model by

1.  **Supernet weight Optimization**<!-- $$\begin{equation} <> \end{equation}$$-->
    :
    $$ \begin{equation}W_{\mathcal{A}}=\mathop{\arg \min}_{W} \: \mathcal{L}_{\text{train}}\big(\mathcal{N}(\mathcal{A},W)\big)\end{equation}$$

    The $$\mathcal{A}$$ is the architecture search space represented as a directed acyclic graph which is encoded as a supernet $$\mathcal{N}(\mathcal{A}, W)$$. During an SGD step in the above equation, each edge in the supernet graph is randomly dropped, using a dropout rate parameter. In this way, the co-adaptation of the node weights is reduced during training making the supernet training easier.

2.  **Architecture Search Optimization**
    :
    $$ \begin{equation}a^* = \mathop{\arg \max}_{a \in \mathcal{A}} \text{ ACC}_{\text{val}}\bigg(\mathcal{N}\big(a,W_{\mathcal{A}}(a)\big)\bigg)\end{equation}$$

    During search, each sampled architecture a inherits its weights from $$W_{\mathcal{A}}$$ as $$W_{\mathcal{A}}(a)$$. The architecture weights are ready to use making the search very efficient and flexible. This type of sequential optimization works because, the accuracy of any architecture $$a$$ on a validation set using inherited weight $$W_{\mathcal{A}}(a)$$ (without extra fine tuning) is highly predictive for the accuracy of $$a$$ that is fully trained. Hence we try to minimize the training loss even further for better performance, supernet weights $$W_{\mathcal{A}}$$ such that all the architectures in the search space are optimized simultaneously.

$$\begin{equation}W_{\mathcal{A}} = \mathop{\arg \min}_{W} \: \mathbb{E}_{a \sim \Gamma(\mathcal{A})}\bigg[\mathcal{L}_{\text{train}}\big(\mathcal{N}(a, W(a))\big)\bigg]\end{equation}$$

where $$\Gamma(\mathcal{A})$$ is the prior distribution of $$a \in \mathcal{A}$$. This stochastic training of the supernet is helpful in better generalization of the optimized model and is also computationally efficient. To overcome the problem of weight coupling, the supernet $$\mathcal{N}(\mathcal{A},W)$$ is chosen such that each architecture is a single path so that this realization is hyperparameter free as compared to traditional NAS approaches. The distribution $$\Gamma(\mathcal{A})$$ is fixed apriori as a uniform distribution during our training and is not a learnable parameter.

## Evaluation and Inference

Evaluation metric is the core for building any accurate machine learning model. We propose to implement the evaluation metrics precision@$$k$$ ,recall@$$k$$ and f1-score@$$k$$ for the augmented sub-models sampled from largest augmented model and for the base model itself, where $$k$$ represents the top k accuracy on the test set. These metrics will assist in evaluating the training procedure better than vanilla accuracy because we are trying to tackle the problem of underfitting, and an underfitted model is more likely to make the same prediction for every input and presence of class imbalance will lead to erroneous results.

## Conclusion

The main conclusion of this blog post is to further refine tiny neural networks effectively without any loss in accuracy and prevent underfitting in them. The paper implements this in a unique way apart from the conventional techniques such as regularization, dropout and data augmentation. This blog adds to other techniques that are orthogonal to NetAug can be used combined with NetAug for improvised results.

<!-- Notes/Overview/Category of the Paper, comment out later
### Category
Solves a known problem of underfitting by proposing a new method

### Context
Regularization, over-fitting, underfitting, data augmentation, model augmentation, training paradigm, TinyML, knowledge distallation, security, privacy

### Correctness
Assumptions appear to be valid, but there are few setbacks (like eq 1 of the paper)

### Contributions
NetAug, Ablation study, improving the performance of underfitting models, new ways to consider loss functions

### Clarity
Well-written overall. Somethings are not clear-->

<!-- Questions/clarifications asked by us -->
<!--```markdown
Hi All,

I hope everyone is doing well. I'm pursuing a BS from India and my friend is an employee. We came across this paper Network Augmentation for Tiny Deep Learning that was published in ICLR 22. We are unclear about a few things, would be great if you can clarify/justify them to us.

1. Why does sampling all augmented networks in one training step decrease the performance, given the training loss increases as one of the reasons?
2. Initially it's mentioned that NetAug is very different from the other knowledge distillation methods later the results show both NetAug and knowledge distillation combined together perform well, we would like to understand it more mathematically so as to why?
3. Things like data loading and communication cost are justified to consume maximum time, but don't we think we can spend training in doing something more potential like maybe in fact learning the feature maps/representations better?

Regards,
Nevasini.
```

```markdown
Hello Han Cai and team,
We are eager to get the above points clarified. Specifically, if you can answer along the following lines, it would be really helpful.
For 1 above: It was mentioned in the paper that it was found sampling more than one augmented network from the maintained single largest augmented model in each training step hurts the performance. We think this counters the intuition and spirit behind the idea of NetAug and the augmented loss proposed in equation (1) of the paper.
For 2 above: Can you further provide insights on why KD and NetAug can be complementary when the underlying motive in NetAug is to solve underfitting and in KD is to generalize well on unseem data with the help of a teacher? These both motives look like they want to solve the generalizability of the model under consideration. Furthermore, KD can be done to tiny models as well right?
For 3 above: We understand that data loading and communication c0st can be a bottleneck in training tiny neural networks. But in the case of NetAug, along with base supervision, we are training the augmented model as well (meaning we update the weights W_i). This means the training cost is expected to increase by 50% and experiments should also show a similar change.

Thanks and regards,
Nevasini and Ipsit.
```

```markdown-->
<!-- Their reply -->
<!--Hi Nevasini and Ipsit,



1. I think it is because the proportion of the base supervision will decrease when we sample more augmented networks, which will make the training process biased toward augmented networks.



2. I think the mechanisms of KD and NetAug are different. KD can be viewed as a better training objective with adaptive label smoothing. It does not encourage the base network to work as a subnetwork of larger networks, different from NetAug.



3. For tiny neural networks, the cost of the additional forward+backward is smaller than you expected. For example, training MobileNetV2-Tiny with NetAug on ImageNet only increases the training time by 16.7% on our GPU servers.



Best,

Han-->
