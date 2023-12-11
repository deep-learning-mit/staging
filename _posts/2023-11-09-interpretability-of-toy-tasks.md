---
layout: distill
title: The Effect of Activation Function On Superposition in Toy Models
description: An in-depth exploration of how different activation functions influence superposition in neural networks.
date: 2023-11-08
htmlwidgets: true
authors:
 - name: Vedang Lad
   url: "https://www.vedanglad.com"
   affiliations:
      name: MIT
 - name: Timothy Kostolansky
   affiliations:
      name: MIT

bibliography: 2023-11-09-interpretability-of-toy-tasks.bib
toc:
  - name: Introduction to Superposition
    subitems:
      - name: Superposition
      - name: Monosemanticity and Polysemanticity
  - name: Problem Setting and Notation
    subitems:
      - name: Problem Specification
      - name: Features
        subitems:
          - name: Sparsity
          - name: Importance
      - name: Dataset
      - name: Network
      - name: Loss
  - name: Results
  - name: AI Safety
  - name: Conclusion

---


### Introduction to Superposition


With the recent emergence of grokking, mechanistic interpretability research has trended towards understanding how models learn \cite{GrokNanda} \cite{Pizza}. A central concept in this pursuit is superposition - a single neuron learning multiple "features."


Features are the distinguishing properties of data points, the “things” that allow a neural network to learn the difference between, say, a dog and a cat, or a Phillip Isola and a Jennifer Aniston. Features are the building blocks that determine what makes one data point different from another. In many cases, features discovered by and encoded within neural networks correspond to human-understandable ideas (sometimes discovering even features and even less) . For example, in language there exist nouns, verbs, and adjectives. It has been found that language models often map these ideas to features within their parameters. Human understanding is not necessary though, as models can find and map features that exist beyond the perception of humans. This is an important part of the success (and dual inscrutability) of modern deep models, as these models can determine features and relationships within the data that allow them to model large datasets, like language, very well.




[2103.01819] The Rediscovery Hypothesis: Language Models Need to Meet Linguistics - the rediscovery hypothesis




**TODO**


**WE should do a "hover" or indented explanation of what features are**


In this work we:


1. Explain Superposition, why it may occur, and why it is important
2. Motivate a framework to easily study Superposition
3. Study how activation functions affect Superposition




### Superposition
Let us elaborate further. If you were to train some neural network and visualize the weights - chances are you would see some mess that looks like this:


**TODO**


You are likely looking at superposition!


As hypothesized by \cite{toymodels}, superposition is a phenomenon which occurs when the number of features being learned by a model is greater than the number of parameters in that model. In order to capture $m$ features with $n<m$ parameters, one can think of the neurons as "working overtime.” In other words, some of the neurons within a model encode information about more than one feature. The neuron exhibiting superposition operates as an information compressor. The caveat is that this compression is often unpredictable and hard to understand!


In a linear model, i.e., one which maps inputs to outputs with only linear functions, there are fewer parameters than the features it tries to represent, so it can only represent the top $n$ features. How then do neural networks use compression and map back to $m>n$ features using only $n$ parameters? The answer is non-linearity. Below we can see how linear and non-linear (using ReLU) networks perform on [SOMETHING, NEED TO EXPLAIN]:


**TODO** Diagram of linear on the left, and RELU on the right


So why do we care about Superposition? Why spend time studying this?


While it may seem tangential, Superposition sheds important insights on Large Language Models (LLMs)! While LLMs are billions of parameters large, this is still not enough for a one-to-one mapping to “features" on the internet. Therefore LLMs also MUST exhibit superposition in order to learn.


**Something here about features on the internet and parameters in a model**


Previous research, as detailed in \cite{toymodels}, has predominantly explored superposition within the confines of toy models utilizing the Rectified Linear Unit (ReLU) activation function \todo[]{TODO cite relu}. However, to extend these findings to contemporary neural networks, it is crucial to investigate the influence of different activation functions on superposition. Different activation functions provide different ways for a model to use superposition to its advantage.


So you train a neural network - what happens at the neuron level?
There are three possibilities. As the network trains each neuron has three choices:


The neuron chooses not to encode the “features”
The neuron chooses to dedicate itself to one feature
The neuron chooses to encode multiple features


The neuron doesn’t choose to do anything - there is no free will - you are born into a loss landscape and an optimizer telling you what to do


In such linear models, each neuron is limited to representing only the most significant features (2), discarding others (1). Conversely, superposition, enabled by non-linear activation functions, adopts a more inclusive approach (3), trying to encode multiple features per neuron and learning efficient representational shortcuts.


While ReLU bears similarity to the Gaussian Error Linear Unit (GeLU) used in modern GPT architectures, a deeper understanding of how different nonlinear activations impact superposition can provide crucial insights. Such understanding is key to unraveling the complex mechanisms through which neural networks utilize non-linearities, a cornerstone in the broader narrative of neural network interpretability.


#### Monosemanticity and Polysemanticity
To connect to existing literature (2) and (3) above are given the names monosemanticity and polysemanticity. We will also follow this notation going forward.


To describe further, the idea of superposition in neural networks leads us to two distinct types of neuron behaviors: monosemanticity and polysemanticity.


Monosemantic neurons are those that specialize in a single, distinct feature, acting as dedicated detectors. This characteristic is often observed in the intermediate layers of architectures like Convolutional Neural Networks (CNNs), where neurons become adept at recognizing specific patterns, such as curves or colors.
Polysemantic neurons do not align with just one feature but engage with multiple features simultaneously, offering a broader and more nuanced understanding of the data. This trait is essential for handling complex, high-dimensional datasets but comes at the cost of reduced interpretability.


### Problem Setting and Notation


Our work extends the work done in \cite{toymodels} by examining how the changing of the activation function on toy model networks affects the behavior and interpretability of these networks. \cite{toymodels} uses the canonical ReLU activation function to add non-linearity to two-layer models in order to analyze how superposition occurs within small networks. Our work substitutes the ReLU function with five other common activation functions: Sigmoid, Tanh, GeLU, SiLU, and SoLU. We hope that generalizing the phenomenon across activation functions can, can push the toy dataset to be in closer to realistic ML settings.


#### Problem Specification


The models in this experiment will be learning how to replicate a length-$n$ vector of inputs in the range $[0, 1]$ with a compression to a length-$m$ embedding (where $n>m$). To set up this experiment, we need to create a dataset that allows for superposition to occur and that also allows for interpretability of the superposition. To motivate this further, we begin with a careful discussion of features.


#### Features


Features are the salient “things” that a neural network learns in order to differentiate inputs.


Technically, features are the properties which neural networks try to extract from data during learning in order to compress inputs to useful representations during inference. Although features can map to human-understandable concepts (e.g., dog ears), features can also map to properties of the data that are not apparent to naive decoding by the human brain. In order to experiment with superposition, we need to encode features in a way that we can understand. In other words, we do not want our experimental model to learn features that we are unaware of. This would make it hard for us to interpret how the model maps features in the data to embeddings within its parameters, consequently obscuring how superposition works. To this aim, we must generate features within the training set for our model which are simple and understandable to us a priori. Similar to \cite{toymodels}, we use as each input a vector with entries drawn independently from a uniform distribution over $[0, 1]$. Making each entry independent of the others enforces that each entry is its own (artificial) feature with no correlation to the other features.


Here we define two important augmentations that we used in the dataset to simulate real-world features: sparsity and importance.


##### Sparsity


Sparsity is a measure of how often a specific feature is present in a dataset. A feature is characterized as “sparse” if it only appears in a fraction of the inputs to the model. Similarly, features that are “dense” appear in many of the inputs.


Specifically, a feature with a sparsity of $S \in [0, 1]$ has a probability $S$ of being expressed in any given input. If we have $S=0$, this means that the feature is expressed in every input, whereas if we have $S=0.5$, this means that the feature is expected to be expressed in about half of the inputs.


In our experiment, we train models at different sparsities in order to capture how sparsity affects superposition.


##### Importance


Not all features are created equal!


Some features are more useful than others in determining relevant information about inputs. For instance, when building a dog detector - capturing features related to dogs’ faces are extremely important! A model would need to pick up salient features of dogs, perhaps floppy ears and snouts. Other features, like the grass a dog is sitting on or a frisbee in a dog’s mouth, may not be as useful for detecting a dog. The differences in usefulness between features is encapsulated in the idea of “importance.”


In the context of feature detection by a neural network, importance plays a role in modulating which features are encoded within the embedded layers of the network. In the context of the superposition hypothesis, if one feature has more importance than another feature, then it would be inefficient for the network to map both features equally within the embedding; allocating more weight to the feature with greater importance would be more valuable to the network in minimizing error.


In our experiment, we give each input feature a different importance in order to allow the models to differentiate between them. We will examine when and how the model justifies mapping multiple features of differing importances to the same neuron, i.e., we will observe the superposition of features with differing importances.


#### Dataset


To run this experiment, we will synthetically generate data that has desired sparsity and importance properties.


Each input $x$ will be a vector of length $n$. Each element $x_i$ in the vector will be drawn independently from the other elements in the uniform range $[0, 1]$. As discussed before, we can now synonymously refer to each of these elements as features, given their independent generation. (We will refer to them as features from this point onwards.)


Each feature $x_i$ in the vector has a relative importance to each of the other features $x_{j\ne i}$. The importance of feature $x_i$ is $I_i = r_I^i$ where $r_I\in(0, 1)$ is a constant describing the relative decay of importance between neighboring features. This attribute of the data will be implemented in the loss function (see below for more details).


We will train separate models for each of the varying levels of sparsity. For an input $x$ with sparsity $S$, each feature $x_i$ will take on its “true” value, a uniformly distributed number, with a probability of $1-S$ and will otherwise be set to 0 with a probability of $S$.


Below is a visualization of two batches of inputs with respective sparsities $S=0.5$ and $S=0.99$.




Caption: Each column of the plots represents a feature vector of length 20. Each batch has size 100, corresponding to the number of columns in the plots. Notice how the changing in sparsity affects the feature density.


#### Network
Below are the architectures of the base (linear) and experimental (non-linear) models that we are using in this experiment. Of particular note is the activation function $\mathbb{f}$, which we will substitute using the aforementioned activation functions.

| Linear Model               | Activation \( $\mathbb{f}$ \) Output Model |
|:---------------------------:|:------------------------------------------:|
| $$h = Wx$$                   | $$h = Wx$$                                   |
| $$ x' = W^T h + b $$   | $$x' = f(W^T h + b)$$          |
| $$x' = W^T Wx + b$$  | $$x' = f(W^T Wx + b)$$          |



**Autoencoder image**


We create an autoencoder - compressing down to induce polysemanticity. This maps $x$ to a direction in a lower-dimensional space, represented by $$h = Wx$$. Each column of $W$ corresponds to a lower-dimensional representation of a feature in $x$. To reconstruct the original vector, $W^T$ is used, ensuring clear feature representation correspondence. This structure results in a symmetric matrix $W^TW$ and allows for clear visualization of the weights. They visually allow for the determination of the presence of superposition.


**EXAMPLE OF WTW**


#### Loss


Sparsity, Importance and Our Network come together in the following loss function:


$$
   L = \sum_{i} \sum_{x} I_{i}(x_{i} - x'_{i})^{2} $$


Motivated by \cite{toymodels}, we use a standard MSE loss, where $x_i$ and $x_i'$ measure the absolute difference in the auto-encoding of the datapoint. The Importance factor, $I_i$ , describes how important the given reconstruction is. A smaller importance will allow loss minimization even with a poor reconstruction.


### Results


Below we present each activation function, along with plots depicting how training results in superposition at varying degrees of sparsity.



[STRUCTURE NOTES]
For each activation function, we should include the Sparsity_super.png, as well as the phase change diagram, at the very least. (Maybe this would be busy/hard to read? Not sure until I see it.)
Should we also include a function definition (ie 1/(1+e^-x) for Sigmoid) and accompanying graph for each (would the additional and likely simple graph be too much visualization)?
Potential orders of activations in paper:
ReLU (OG), Sigmoid/Tanh (simple, OG, similar), GeLU/SiLU (still per-neuron, good (?) transition to SoLU), SoLU
ReLU (OG), GeLU/SiLU (similar results: antipodal pairs), Sigmoid/Tanh (more polysemanticity), SoLU (multi-neuron activation),
Similar activation shapes: ReLU/GeLU/SiLU (flat + linear-ish), Sigmoid/Tanh (S), SoLU (indie boy)


The ReLU (Rectified Linear Units) activation function is a piecewise-linear function, a property which results in the ReLU model attempting to simulate the identity function when possible. [IDK WE NEED MORE AND IDK, look at comment]
Similar to the results in \cite{toymodels}, the ReLU model focuses on the most significant features in the low sparsity regime (generally resulting in monosemanticity), while relying on superposition in the high sparsity regime (polysemanticity). With weaker signals for the most important features in the high sparsity regime, the model encodes multiple features in each neuron activation in order to minimize error of the sparse signals. \cite{toymodels} explores this model extensively and gives a baseline for how superposition appears in toy models. Notably, the ReLU model uses antipodal pairs in the mapping of features in order to encode multiple features to single neurons. This can be seen as a light-colored diagonal entry within $W^T W$ and a corresponding dark-colored off-diagonal entry within the same column. This antipodal mapping of features is a method that the model uses to compress more than one feature to one neuron. This antipodal mapping is more interpretable than other kinds of polysemanticity which occurs in subsequently-described activation functions which “speckle” multiple features into a single neuron, making it more difficult to determine how the superposition occurs in that model.
[SOMETHING ABOUT THE DRAWbACK OF RELU MODEL PERHAPS]


GeLU (Gaussian Error Linear Units)
GeLU have recently been popularized as they are the activation functions of choice in many transformers, including BERT and GPT. The GeLU is differentiable for all $x$ - and has a smoother curve than the SilU (Swish) activation. \cite{elhage2022solu} found that in the setting of transformers, the GeLU was less interpretable than the Solu. This may be the case after having many linear layers activation - but a single layer this is not the case.


SiLU (Sigmoid Linear Units)


The GeLU and SiLU models exhibit similar kinds of superposition in their weight matrices. With increasing sparsity, superposition of features does happen, but it is more “strict” than the ReLU model, perceptibly mapping at most two features to any single neuron. In all polysemantic neurons, though, there is one feature that dominates, suggesting that these activation functions enforce sparsity in their activations. There are also many antipodal pairs of features within these models, reiterating the behavior that exists in the ReLU models (also found in \cite{toymodels}).


The Sigmoid function is a smooth activation function with an output range of $(0, 1)$. This maps directly to the desired range of values that the model is trying to replicate. The Sigmoid model exhibits superposition in all neurons as soon as the  sparsity is non-zero, as can be seen from the “speckling” of non-zero off-diagonal terms in $W^T W$. This is a difference from the ReLU model, for which the superposition “leaks” into the least significant encoded features at low, non-zero sparsities and eventually affects all features at higher sparsities. This low-sparsity superposition may occur because the Sigmoid function strictly maps to $(0, 1)$, with increasingly large pre-activation inputs necessary to map to values close to 0 and 1. As such, the model may be “speckling” the off-diagonal values in an attempt to “reach” these inputs which are close to 0 and 1.




The Tanh function is another smooth activation function, but it results in significantly different behavior, prioritizing the most important features regardless of sparsity. This behavior is possibly attributed to the range that the Tanh function maps to: $(-1, 1)$.


Despite similarities in the S-like curvature of the Sigmoid and Tanh activation functions, the Sigmoid model exhibits superposition, whereas the Tanh model exhibits nearly zero superposition. One key difference between the two functions is the fact that the Sigmoid function maps inputs to a range of $(0, 1)$, while the Tanh function maps inputs to a range of $(-1, 1)$. This difference is significant in our experiment, as our experiment uses models to recreate random vectors with elements in the range $[0, 1]$. The range of the Sigmoid function matches this range, while the range of the Tanh function which matches this range only occurs for non-negative inputs to the Tanh function. In other words, the $(-\infty, 0)$ input domain (which maps to the range $(-1, 0)$) of the Tanh function remains useless for prediction of values which should be in the range $[0, 1]$. Therefore, the tanh function empirically acts like a linear function (i.e., no activation layer).


The SoLU (Softmax Linear Units) activation function is based on the work from \cite{elhage2022solu}. SoLU is a function for which the activation of each neuron is dependent on all the other neurons within its own layer. This is significantly different from the other activations that we tested, as the activations of neurons with the other functions are independent of the other neurons within the same layer. In other words, the other activation functions are univariate while the SoLU is multivariate. Similar to other approaches like L1 regularization, the SoLU amplifies neurons with relatively large pre-activations and de-amplifies neurons with relatively smaller pre-activations. This behavior pressures the model to be more monosemantic (and therefore more interpretable in some settings), as discussed in \cite{elhage2022solu}. In our experiment, the SoLU model results in non-zero superposition of all features with all amounts of sparsity. This may be attributed to the way that the SoLU “forces” activations to be sparse, i.e., the activations result in a “winner-takes-all” behavior due to the way that the Softmax function works. This is not a useful property for prediction of a vector of independently-drawn values, as the input vectors are unlikely to be peaky, i.e., the SoLU does not quite fit the purposes of its task.


SoLU blog citation:
@article{elhage2022solu,
   title={Softmax Linear Units},
   author={Elhage, Nelson and Hume, Tristan and Olsson, Catherine and Nanda, Neel and Henighan, Tom and Johnston, Scott and ElShowk, Sheer and Joseph, Nicholas and DasSarma, Nova and Mann, Ben and Hernandez, Danny and Askell, Amanda and Ndousse, Kamal and Jones, Andy and Drain, Dawn and Chen, Anna and Bai, Yuntao and Ganguli, Deep and Lovitt, Liane and Hatfield-Dodds, Zac and Kernion, Jackson and Conerly, Tom and Kravec, Shauna and Fort, Stanislav and Kadavath, Saurav and Jacobson, Josh and Tran-Johnson, Eli and Kaplan, Jared and Clark, Jack and Brown, Tom and McCandlish, Sam and Amodei, Dario and Olah, Christopher},
   year={2022},
   journal={Transformer Circuits Thread},
   note={https://transformer-circuits.pub/2022/solu/index.html}
}




### AI Safety 
Interpretable models would be a useful tool for safe AI systems. It would be easy to understand why models made the decision they did and how they made it. As presented by \cite{elhage2022solu}, the SoLU pushes models to mono semanticity in the transformer architecture, allowing increasing interpretability - however this forces some parts of the model into polysemanticity.


Interpretability is not free!


There is a prevailing concern that models labeled as 'interpretable' can be misleading. This stems from the notion that if a model is only partially transparent, revealing some aspects while concealing others, it could potentially lead to deception. i.e hiding elements through poly  semanticity.


### Conclusion


Our exploration into the realm of activation functions and their influence on superposition in neural networks goes beyond academic curiosity. It opens up new avenues in making neural network models more interpretable, transparent, and secure – aligning closely with the ever-important field of AI safety.


{% bibliography --cited %}

