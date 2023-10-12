---
layout: distill
title: How much meta-learning is in image-to-image translation?
description: ...in which we find a connection between meta-learning literature and a paper studying how well CNNs deal with nuisance transforms in a class-imbalanced setting. Closer inspection reveals a surprising amount of similarity - from meta-information to loss functions.
date: 2022-12-01
htmlwidgets: true

# anonymize when submitting 
authors:
  - name: Anonymous 

# do not fill this in until your post is accepted and you're publishing your camera-ready post!
# authors:
#   - name: Albert Einstein
#     url: "https://en.wikipedia.org/wiki/Albert_Einstein"
#     affiliations:
#       name: IAS, Princeton
#   - name: Boris Podolsky
#     url: "https://en.wikipedia.org/wiki/Boris_Podolsky"
#     affiliations:
#       name: IAS, Princeton
#   - name: Nathan Rosen
#     url: "https://en.wikipedia.org/wiki/Nathan_Rosen"
#     affiliations:
#       name: IAS, Princeton 

# must be the exact same name as your blogpost
bibliography: 2022-12-01-how-much-meta-learning-is-in-image-to-image-translation.bib  

# Add a table of contents to your post.
#   - make sure that TOC names match the actual section names
#     for hyperlinks within the post to work correctly.
toc:
  - name: A closer look at the experiment
  - name: How is this a meta-learning experiment? 
  - name: Generative Invariance Transfer
  - name: How much meta-learning is in MUNIT?
  - name: Conclusion
---

At the last ICLR conference, Zhou et al. [2022] <d-cite key="DBLP:conf/iclr/ZhouTRKPHF22"></d-cite> presented work showing that CNNs do not transfer information between classes of a classification task. 

- Allan Zhou, Fahim Tajwar, Alexander Robey, Tom Knowles, George J. Pappas, Hamed Hassani, Chelsea Finn [ICLR, 2022] Do Deep Networks Transfer Invariances Across Classes?<d-cite key="DBLP:conf/iclr/ZhouTRKPHF22"></d-cite>

Here is a quick summary of their findings: 
If we train a Convolutional Neural Net (CNN) to classify fruit on a set of randomly brightened and darkened images of apples and oranges, it will learn to ignore the scene's brightness. We say that the CNN learned that classification is **invariant** to the **nuisance transformation** of randomly changing the brightness of an image. We now add a set of plums to the training data, but fewer examples of them than we have apples and oranges. However, we keep using the same random transformations. The training set thus becomes **class-imbalanced**.

We might expect a sophisticated learner to look at the entire dataset, recognize the random brightness modifications across all types of fruit and henceforth ignore brightness when making predictions. If this applied to our fruit experiment, the CNN would be similarly good at ignoring lighting variations on all types of fruit. Furthermore, we would expect the CNN to become more competent at ignoring lighting variations in proportion to **the total amount of images**, irrespective of which fruit they depict. 

Zhou et al. [2022] <d-cite key="DBLP:conf/iclr/ZhouTRKPHF22"></d-cite> show that a CNN does not behave like this: When using a CNN on a **class-imbalanced** classification task with random nuisance transformations, the CNNs invariance to the transformation is proportional to the size of the training set **for each class**. This finding suggests CNNs don't **transfer invariance** between classes when learning such a classification task.

However, there is a solution: Zhou et al. [2022] <d-cite key="DBLP:conf/iclr/ZhouTRKPHF22"></d-cite> use an Image to Image translation architecture called MUNIT<d-cite key="DBLP:conf/eccv/HuangLBK18"></d-cite> to learn the transformations and generate additional data from which the CNN can learn the invariance separately for each class. Thus, the invariance to nuisance transformations is transferred **generatively**. They call this method **Generative Invariance Transfer (GIT)**.

In this blog post, we are going to argue that:
- The experiment described above is a meta-learning experiment.
- MUNIT is related to meta-learning methods.

Before we proceed to the main post, let's clarify some definitions. If you are already familiar with the subject, you may skip this part:

<details>
  <summary><b> Definition: Class-Imbalanced Classification</b></summary>
  <br/>
  <p>
     In many real-world classification datasets, the number of examples for each class varies. Class-imbalanced classification refers to classification on datasets where the frequencies of class labels vary significantly. 
  </p>
  <p>
    It is generally more difficult for a neural network to learn to classify classes with fewer examples. However, it is often important to perform well on all classes, regardless of their frequency in the dataset. If we train a model to classify a dataset of different skin tumors, most examples may be benign. Still, it is crucial to identify the rare, malignant ones. Experiment design, including training and evaluation methods must therefore be adjusted when using class-imbalanced data.
    </p>
    <br/>
</details>
<details>
  <summary><b> Definition: Nuisance Transformation & Transformation Invariance</b></summary>
  <br/>
  <p>
    Transformations are alterations of data. In the context of image classification, nuisance transformations are alterations that do not affect the class labels of the data. A model is said to be invariant to a nuisance transformation if it can successfully ignore the transformation when predicting a class label.
  </p>
    We can formally define a nuisance transformation
  <p>
    $$T(\cdot |x)$$
  </p>
  <p>
    as a distribution over transformation functions. An example of a nuisance transformation might be a distribution over rotation matrices of different angles, or lighting transformations with different exposure values. By definition, nuisance transformations have no impact on class labels $y$, only on data $x$. A perfectly transformation-invariant classifier would thus completely ignore them, i.e.,
  </p>
  <p>
    $$
        \hat{P}_w(y = j|x) = \hat{P}_w(y = j|x'), \; x' \sim T(\cdot |x).
    $$
  </p>
</details>


## A closer look at the experiment

Let's take a more detailed look at the experiment Zhou et al. [2022] <d-cite key="DBLP:conf/iclr/ZhouTRKPHF22"></d-cite> conducted:

Zhou et al. [2022] <d-cite key="DBLP:conf/iclr/ZhouTRKPHF22"></d-cite> take a dataset, e.g., CIFAR-100, then apply a nuisance transformation, for example, random rotation, background intensity, or dilation and erosion. They then remove samples from some classes until the distribution of class sizes follows Zipf's law with parameter 2.0 and a minimum class size of 5. The test set remains balanced, i.e., all test classes have the same number of samples. They then train a CNN model - for example, a ResNet - on this imbalanced and transformed training data. 

To measure the invariance of the trained model to the applied transformation Zhou et al. [2022] <d-cite key="DBLP:conf/iclr/ZhouTRKPHF22"></d-cite> use the empirical Kullback-Leibler divergence between the untransformed test set and the transformed test set of each class. 

<p>
$$
    eKLD(\hat{P}_w) = \mathbb{E}_{x \sim \mathbb{P}_{test}, x' \sim T(\cdot|x)} [D_{KL}(\hat{P}_w(y = j|x) || \hat{P}_w(y = j|x'))]
$$
</p>

If the learner is invariant to the transformation, the predicted probability distribution over class labels should be identical for the transformed and untransformed images. In that case, the KLD should be zero and greater than zero otherwise. The higher the expected KL-divergence, the more the applied transformation impacts the network's predictions.

The result: eKLD falls with class size. This implies that the CNN does not learn that there are the same nuisance transformations on all images and therefore does not transfer this knowledge to the classes with less training data. A CNN learns invariance **separately for each class**.

{% include figure.html path="assets/img/2022-12-01-how-much-meta-learning-is-in-image-to-image-translation/EKLD.svg" class="img-fluid" %}


## How is this a meta-learning experiment? 

You might think this is a cool experiment, but how is it related to meta-learning? 

Let's look at one of the original papers on meta-learning. In the 1998 book "Learning to learn" Sebastian Thrun & Lorien Pratt define an algorithm as capable of "Learning to learn" if it improves its performance in proportion to the number of tasks it is exposed to:

>an algorithm is said to learn to learn if its performance at each task improves with experience and with the number of tasks. Put differently, a learning algorithm whose performance does not depend on the number of learning tasks, which hence would not benefit from the presence of other learning tasks, is not said to learn to learn <d-cite key="DBLP:books/sp/98/ThrunP98"></d-cite>

Now how does this apply to the experiment just outlined? In the introduction, we thought about how a sophisticated learner might handle a dataset like the one described in the last section. We said that a sophisticated learner would learn that the nuisance transformations are applied uniformly **to all classes**. Therefore, if we added more classes to the dataset, the learner would become **more invariant** to the transformations because we expose it to more examples of them. Since this is part of the classification task **for each class**, the learner should, everything else being equal, become better at classification, especially on classes with few training examples. To see this, we must think of the multi-classification task not as a single task but as multiple mappings from image features to activations that must be learned, as a set of binary classification tasks. Thrun and Pratt continue:

>For an algorithm to fit this definition, some kind of *transfer* must occur between multiple tasks that must have a positive impact on expected task-performance <d-cite key="DBLP:books/sp/98/ThrunP98"></d-cite>.

This transfer is what Zhou et al. [2022] <d-cite key="DBLP:conf/iclr/ZhouTRKPHF22"></d-cite> tried to measure. There is some meta-information learnable across several tasks, in our case, the transformation distribution across many binary classification tasks. If a learner can learn this meta-information and transfer it to each new task it has "learned to learn"; it is a meta-learner. The goal of Zhou et al.'s <d-cite key="DBLP:conf/iclr/ZhouTRKPHF22"></d-cite> experiment was to see whether this transfer takes place. Thus, arguably, it is a meta-learning experiment.

## Generative Invariance Transfer

Zhou et al. [2022] <d-cite key="DBLP:conf/iclr/ZhouTRKPHF22"></d-cite> don't stop there. They show that, using the MUNIT (Multimodal Unsupervised image-to-image Translation)<d-cite key="DBLP:conf/eccv/HuangLBK18"></d-cite>  architecture, they can learn the nuisance transformations applied to the dataset and generate additional training samples for the classes with few samples, improving transformation invariance there. They call this Generative invariance transfer (GIT). Let's take a closer look: 

MUNIT networks are capable of performing image-to-image translation, which means that they can translate an image from one domain, such as pictures of leopards, into another domain, such as pictures of house cats. The translated image should look like a real house cat while still resembling the original leopard image. For instance, if the leopard in the original image has its eyes closed, the translated image should contain a house cat with closed eyes. Eye state is a feature present in both domains, so a good translator should not alter it. On the other hand, a leopard's fur is yellow and spotted, while a house cat's fur can be white, black, grey, or brown. To make the translated images indistinguishable from real house cats, the translator must thus replace leopard fur with house cat fur.

{% include figure.html path="assets/img/2022-12-01-how-much-meta-learning-is-in-image-to-image-translation/MUNIT_ENCODING.svg" class="img-fluid" %}

MUNIT networks learn to perform translations by correctly distinguishing the domain-agnostic features (such as eye state) from the domain-specific features (such as the distribution of fur color). They embed an image into two latent spaces: a content space that encodes the domain-agnostic features and a style space that encodes the domain-specific features (see figure above).

To transform a leopard into a house cat, we can encode the leopard into a content and a style code, discard the leopard-specific style code, randomly select a cat-specific style code, and assemble a house cat image that looks similar by combining the leopard's content code with the randomly chosen cat style code (see figure below).

{% include figure.html path="assets/img/2022-12-01-how-much-meta-learning-is-in-image-to-image-translation/MUNIT_TRANSLATION.svg" class="img-fluid" %}

Zhou et al. [2022] <d-cite key="DBLP:conf/iclr/ZhouTRKPHF22"></d-cite> modify the process of using MUNIT to transfer images between domains. They do not use MUNIT to translate images **between** domains but **within** a domain. The MUNIT network exchanges the style code of an image with another style code of the same domain. For example, if the domain is house cats, the MUNIT network might translate a grey house cat into a black one. The learning task in this single-domain application of MUNIT is to decompose example-agnostic content features from example-specific style features so that the translated images still look like house cats. For example, fur color is a valid style feature for translating within the 'house cat' domain because every house cat has a fur color. A translator only switching fur color is hard to detect.

 However, if the domain included house cats **and apples**, fur color is not a valid style feature. If it was, the translator might translate fur color on an apple and give it black fur, which would look suspiciously out of place. Whatever house cats and apples have in common - maybe their position or size in the frame - would be a valid style feature. We would expect an intra-domain translator on an apples-and-cats dataset to change the position and size of an apple but not to turn it into a cat (not even partially).

It turns out that on a dataset with uniformly applied nuisance transformations, the nuisance transformations are valid style features: The result of randomly rotating an apple cannot be discerned as artificial when images of all classes, house cats and apples, were previously randomly rotated. 

Zhou et al. [2022] <d-cite key="DBLP:conf/iclr/ZhouTRKPHF22"></d-cite> find that when they train a MUNIT network on a dataset with nuisance transformations and class imbalances, the MUNIT network decomposes the class and transformation distributions. The style latent space of the MUNIT network approximates the transformation distribution $T(\cdot &#124;x)$. The content space preserves the remaining features of the image, such as its class. Thus, when translating an image, i.e., exchanging its style code, MUNIT applies a random nuisance transformation while preserving content. Zhou et al. [2022] <d-cite key="DBLP:conf/iclr/ZhouTRKPHF22"></d-cite> use this method to generate data for classes with few examples. While the CNN is still unable to transfer invariance to $T(\cdot &#124;x)$ between classes, it can now learn it for each class separately using the data generated by MUNIT, which has acquired knowledge of $T(\cdot &#124;x)$ from the entire dataset.

So MUNIT can decompose the example-specific information, e.g., whether something is an apple or a house cat, from the meta-information, the applied nuisance transformations. When we add more classes, it has more data and can better learn the transformation distribution T(\cdot &#124;x)$. Does solving a meta-learning problem make MUNIT a meta-learner? Let's look at the relationship MUNIT has with traditional meta-learners.

## How much meta-learning is in MUNIT?

To see how well MUNIT fits the definition of meta-learning, let's define meta-learning more concretely. We define contemporary neural-network-based meta-learners in terms of a learning procedure: An outer training loop with a set of trainable parameters iterates over tasks in a  distribution of tasks. Formally a task is comprised of a dataset and a loss function $ \mathcal{T} = \\\{ \mathcal{D}, \mathcal{L} \\\} $. In an inner loop, a learning algorithm based on the outer loop's parameters is instantiated for each task. We train it on a training set (*meta-training*) and test it on a validation set (*meta-validation*). We then use loss on this validation set to update the outer loop's parameters. In this task-centered view of meta-learning, we can express the objective function as

<p>
$$
\underset{\omega}{\mathrm{min}} \; \mathbb{E}_{\mathcal{T} \sim p(\mathcal{T})} \; \mathcal{L}(\mathcal{D}, \omega), 
$$
</p>

where $ \omega $ is parameters trained exclusively on the meta-level, i.e., the *meta-knowledge* learnable from the task distribution <d-cite key="DBLP:journals/pami/HospedalesAMS22"></d-cite>.

This *meta-knowledge* is what the meta-learner accumulates and transfers across the tasks in Thrun and Pratt's definition above. Collecting meta-knowledge allows the meta-learner to improve its expected task performance with the number of tasks. The meta-knowledge in the experiment of Zhou et al. [2022] <d-cite key="DBLP:conf/iclr/ZhouTRKPHF22"></d-cite> is the invariance to the nuisance transformations as the transformations are identical and need to be ignored for images of all classes. By creating additional transformed samples, the MUNIT network makes the meta-knowledge learnable for the CNN.

The task-centered view of meta-learning brings us to a related issue: A meta-learner must discern and decompose task-specific knowledge from meta-knowledge. Contemporary meta-learners decompose meta-knowledge through the different objectives of their inner and outer loops and their respective loss terms. They store meta-knowledge in the outer loop's parameter set $ \omega $ but must not learn task-specific information there. Any unlearned meta-features lead to slower adaptation, negatively impacting performance, *meta-underfitting*. On the other hand, any learned task-specific features will not generalize to unseen tasks in the distribution, thus also negatively impacting performance, *meta-overfitting*.

We recall that, similarly, MUNIT <d-cite key="DBLP:conf/eccv/HuangLBK18"></d-cite> decomposes domain-specific style information and domain-agnostic content information. Applied to two domains, leopards and house cats, a MUNIT network will encode the domain-agnostic information, e.g., posture, scale, background, in its content latent space, and the domain-specific information, e.g., how a cat's hair looks, in its style latent space. If the MUNIT network encoded the domain-agnostic information in the style latent space, the resulting image would not appear to be a good translation since the style information is discarded and replaced. It might turn a closed-eyed leopard into a staring cat. If the MUNIT network encoded the domain-specific transformation in the content latent space, the network would have difficulty translating between domains. A house cat might still have its original leopard fur.

Both meta-learning and multi-domain unsupervised image-to-image translation are thus learning problems that require a separation of the general from the specific. This is even visible when comparing their formalizations as optimization problems.

Francheschi et al. [2018] <d-cite key="DBLP:conf/icml/FranceschiFSGP18"></d-cite> show that all contemporary neural-network-based meta-learning approaches can be expressed as bi-level optimization problems. We can formally write the optimization objective of a general meta-learner as:

<p>
$$
\bbox[5pt, border: 2px solid blue]{
\begin{align*}
   \omega^{*} = \underset{\omega}{\mathrm{argmin}} \sum_{i=1}^{M} \mathcal{L}^{meta}(\theta^{* \; (i)}(\omega), D^{val}_i),
\end{align*}
}
$$
</p>


where $M$ describes the number of tasks in a batch, $\mathcal{L}^{meta}$ is the meta-loss function, and $ D^{val}_i $ is the validation set of the task $ i $. $\omega$ represents the parameters exclusively updated in the outer loop. $ \theta^{* \; (i)} $ represents an inner loop learning a task that we can formally express as a sub-objective constraining the primary objective

<p>
$$
\bbox[5pt, border: 2px solid red]{
\begin{align*}
   s.t. \; \theta^{* \; (i)} = \underset{\theta}{\mathrm{argmin}} \; \mathcal{L^{task}}(\theta, \omega, D^{tr}_i),
\end{align*}
}
$$
</p>

where $ \theta $ are the model parameters updated in the inner loop, $ \mathcal{L}^{task} $ is the loss function by which they are updated and $ D^{tr}_i $ is the training set of the task $ i $  <d-cite key="DBLP:journals/pami/HospedalesAMS22"></d-cite>.

It turns out that the loss functions of MUNIT can be similarly decomposed:

{% include figure.html path="assets/img/2022-12-01-how-much-meta-learning-is-in-image-to-image-translation/MUNIT_LOSS.svg" class="img-fluid" %}


MUNIT's loss function consists of two adversarial (GAN) <d-cite key="DBLP:conf/nips/GoodfellowPMXWOCB14"></d-cite> loss terms (see figure above) with several auxiliary reconstruction loss terms. To keep the notation simple, we combine all reconstruction terms into a joined reconstruction loss $ \mathcal{L}_{recon}(\theta_c, \theta_s) $, where $ \theta_c $ are the parameters of the *content* encoding/decoding networks and $ \theta_s $ are the parameters of the *style* encoding/decoding networks. We will only look at one of the two GAN losses in detail since they are symmetric, and one is discarded entirely when MUNIT is used on a single domain in the fashion of Zhou et al. [2022] <d-cite key="DBLP:conf/iclr/ZhouTRKPHF22"></d-cite>.

MUNIT's GAN loss term is

<p>
$$
\begin{align*}
    &\mathcal{L}^{x_{2}}_{GAN}(\theta_d, \theta_c, \theta_s) 
    \\\\
    =& \;\mathbb{E}_{c_{1} \sim p(c_{1}), s_{2} \sim p(s_{2})} \left[ \log (1 -D_ {2} (G_{2} (c_{1}, s_{2}, \theta_c, \theta_s), \theta_d)) \right]
    \\
    +& \;\mathbb{E}_{x_{2} \sim p(x_{2})}  \left[ \log(D_{2} (x_{2}, \theta_d)) \right],
\end{align*}
$$
</p>

where the $ \theta_d $ represents the parameters of the discriminator network, $p(x_2)$ is the data of the second domain, $ c_1 $ is the content embedding of an image from the first domain to be translated. $ s_2 $ is a random style code of the second domain. $ D_2 $ is the discriminator of the second domain, and $ G_2 $ is its generator. MUNIT's full objective function is:

<p>
$$
\begin{align*}
        \underset{\theta_c, \theta_s}{\mathrm{argmin}} \; \underset{\theta_d}{\mathrm{argmax}}& \;\mathbb{E}_{c_{1} \sim p(c_{1}), s_{2} \sim p(s_{2})} \left[ \log (1 -D_ {2} (G_{2} (c_{1}, s_{2}, \theta_c, \theta_s), \theta_d)) \right]
    \\ +& \; \mathbb{E}_{x_{2} \sim p(x_{2})}  \left[ \log(D_{2} (x_{2}, \theta_d)) \right], + \; \mathcal{L}^{x_{1}}_{GAN}(\theta_d, \theta_c, \theta_s) 
    \\ +& \;\mathcal{L}_{recon}(\theta_c, \theta_s)
\end{align*}
$$
</p>

(compare <d-cite key="DBLP:conf/eccv/HuangLBK18, DBLP:conf/nips/GoodfellowPMXWOCB14"></d-cite>).
We can reformulate this into a bi-level optimization problem by extracting a minimization problem describing the update of the generative networks.
We also drop the second GAN loss term as it is not relevant to our analysis. 

<p>
$$
\bbox[5px, border: 2px solid blue]{
\begin{align*}
    \omega^{*} 
    & = \{ \theta_c^*, \theta_s^* \} 
    \\\\
    & = 
    \underset{\theta_c, \theta_s}{\mathrm{argmin}} \; \mathbb{E}_{c_{1} \sim p(c_{1}), s_{2} \sim p(s_{2})} \left[ \log (1 -D_ {2} (G_{2} (c_{1}, s_{2}, \theta_c, \theta_s), \theta_d^{*})) \right]
    \\
    & + \mathcal{L}_{recon}(\theta_c, \theta_s),
\end{align*}
}
$$
</p>

We then add a single constraint, a subsidiary maximization problem for the discriminator function:

<p>
$$
\bbox[5px, border: 2px solid red]{
\begin{align*}
   &s.t. \;\theta_d^{*}
   \\\\
    & =
    \underset{\theta_d}{\mathrm{argmax}} \; \mathbb{E}_{c_{1} \sim p(c_{1}), s_{2} \sim p(s_{2})} \left[ \log (1 -D_ {2} (G_{2} (c_{1}, s_{2}, \theta_c, \theta_s), \theta_d)) \right] 
    \\
    & + \mathbb{E}_{x_{2} \sim p(x_{2})}  \left[ \log(D_{2} (x_{2}, \theta_d)) \right]
\end{align*}
}
$$
</p>


Interestingly, this bi-level view does not only resemble a meta-learning procedure as expressed above, but the bi-level optimization also facilitates a similar effect. Maximizing the discriminator's performance in the constraint punishes style information encoded as content information. If style information is encoded as content information, the discriminator detects artifacts of the original domain in the translated image. Similarly, a meta-learner prevents *meta-overfitting* via an outer optimization loop. 

The two procedures also share "indirect" parameter updates. During GAN training, the discriminator's parameters are updated through the changes in the generator's parameters, which derive from the discriminator's previous parameters, and so forth; The training of the discriminator and generator are separate but dependent processes. Similarly, in a meta-learner, the outer loop impacts the inner loop by determining its initial learner. The results of the inner loop, meanwhile, impact the outer loop via the meta-validation loss. While often overlapping in practice, the inner and outer loop parameter sets could, in principle, be completely disjunct, as they are in GAN training.

Concluding, we discern between MUNIT applied to two domains versus a single domain. When applied to two domains, MUNIT is a *binary* image-to-image translation architecture, i.e., we cannot add domains. For multi-domain translation, we need to train many MUNIT networks to translate between pairs of domains. Thus, although it uses mechanisms similar to a meta-learner, it is *not* a meta-learner in an image-to-image translation context. 

However, when applied to a single domain MUNIT *does* "learn to learn" (if you agree with the conclusion of the previous chapter) as it combines information from all classes to extract the transformation distribution. While it does not *perform* classification, the class information of an image is encoded in MUNIT's content space. Since MUNIT is trained unsupervised, it is probably closer to a distance metric than an actual class label. We might thus classify single-domain MUNIT as an unsupervised, generative meta-learner. It performs meta-learning in a general sense of "discerning the general from the task-specific" using a related two-loop training approach with two sets of parameters and a similar loss function. Thus MUNIT is related to meta-learning methods.

## Conclusion

The ICLR paper "Do Deep Networks Transfer Invariances Across Classes?" <d-cite key="DBLP:conf/iclr/ZhouTRKPHF22"></d-cite> shows that image-to-image translation methods can be used to learn and apply nuisance transformations, enabling a CNN to become invariant to them via data augmentation. This blog post argued that this is a meta-learning setting. In the view of this author, Zhou et al. [2022] <d-cite key="DBLP:conf/iclr/ZhouTRKPHF22"></d-cite> solve a meta-learning problem using an unsupervised, generative method. A closer examination reveals parallels between both types of architecture. 

*Learning the meta-information of image-to-image translation and meta-learning might enable researchers to design better architectures in both domains.*
