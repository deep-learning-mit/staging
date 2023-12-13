---
layout: distill
title: 'Optimizations of Transformers for Small-scale Performance'
description: 'CNNs generally outperform ViTs in scenarios with limited training data. However, the narrative switches when the available training data is extensive. To bridge this gap and improve upon existing ViT methods, we explore how we can leverage recent progress in the transformer block and exploit the known structure of pre-trained ViTs.'
date: 2023-11-15
htmlwidgets: true

authors:
  - name: Sebastian Diaz
    url:
    affiliations:
      name: MIT

# must be the exact same name as your blogpost
bibliography: 2023-11-08-diaz-proposal.bib  

toc:
  - name: 'Transformers: Great But Not Enough'
    subsections:
      - name: Basic Background
      - name: 'Vision: The Problem'
      - name: Transformer Block
  - name: 'Translation to Vision: Experimentation and Analysis'
    subsections:
      - name: Vanilla vs. Simplified Comparison
      - name: Initialization Schemes
  - name: 'Conclusion and Limitations'

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

<div class="text-center">
   <a name="figure-1"></a> 
  <img src="https://discuss.tensorflow.org/uploads/default/original/2X/4/44b54935a57a92b71902d81265e9bc3c6d99fb12.gif" width="700" height="250">

  <p class="caption">
    Figure 1: Attention Maps of a Vision Transformer (DINO). Source: <a href="https://github.com/sayakpaul/probing-vits">https://github.com/sayakpaul/probing-vits </a>.
  </p>
</div>

## Transformers: Great but fall short
### Basic Background
Transformers have well-earned their place in deep learning. Since the architecture's introduction in<d-cite key="AttentionIsAllYouNeed"></d-cite>, we have seen huge improvements in our model's capabilities. The most notable of which being natural language processing (NLP) with large-language models such as GPT-4 stunning the world at-large.

Originally designed for NLP, the transformer architecture has been robust in other domains and tasks. For example, it has been translated, with success, to de-novo protein design<d-cite key="Grechishnikova2021"></d-cite>, the medical field<d-cite key = "Hu2022"></d-cite>, and, of most relevance, computer vision<d-cite key = "Dosovitskiy2020"></d-cite>. This behaviour differs from architectures of the past like RNNs and CNNs which have been limited to one domain. The potent generalizability of the transformer lies within the self-attention mechanism. Without getting to much into detail, self-attention enables nodes within a neural network to probe the input sequence, determine what is most interesting, and attend towards the region of interest by dynamically updating its weights. Visualization of attention can be seen in [Figure 1](#figure-1). By probing the data landscape, the architecture enables long-range dependencies to be modeled regardless of distance. From a Fourier perspective, the transformer caters towards the low-frequency information in the data and deciphers how each element of an input sequence all relate to each other<d-cite key="Wang2022"></d-cite>. These connections help the transformer accurately model global information in the data perhaps indicating why they are so powerful. In this blog, we will specifically examine the transformer in vision, determine how it can be improved, and evaluate new strategies to increase its viability on small datasets.

<div class="col-sm text-center">
  <a name="figure-2"></a> 
  {% include figure.html path="assets/img/2023-11-08-diaz-proposal/vit_workflow.png" class="img-fluid rounded z-depth-1" %}
  <div class="caption"> 
    Figure 2: ViT workflow.
  </div>
</div>


### Vision: The Problem
The Vision Transformer (ViT)<d-cite key = "Dosovitskiy2020"></d-cite> introduced the transformer to the computer vision world in late 2020. The ViT is simple: it funnels image patches into a tokenization scheme, adds positional encoding, and feeds these tokens into a transformer block. A graphical workflow of the ViT from the original paper can be seen in [Figure 2](#figure-2).

Since its introduction, the ViT and associated variants have demonstrated remarkable benchmarks in image classification<d-cite key = "Liu2021"></d-cite>, image restoration<d-cite key = "Liang2021"></d-cite>, and object detection<d-cite key = "Li2022"></d-cite>. Much of these new methods can compete and even outperform long-established CNNs. However, ViTs are data-hungry requiring extensive amounts of training data to surpass CNNs. In small scale training, ViTs are burdensome to train and achieve sub-par performance compared to their CNNs counterparts<d-cite key ="Naimi2021"></d-cite>. In <d-cite key = "Zhu2023"></d-cite>, they investigate this discrepancy by comparing the feature and attention maps of small-scale CNNs and ViTs, respectively. The authors determine the ViT lacks the ability to learn local information and has ill-suited representation capacity in the lower layers. In contrast, the CNN demonstrate remarkable inductive bias due to weight sharing and locality properties which enable high-frequency modeling<d-cite key = "Park2022"></d-cite>. The ViT's low-frequency and the CNNs high-frequency capacity has initiated a wave of new models aimed at combining the two for comprehensive modeling capability<d-cite key = "Si2022"></d-cite><d-cite key = "ConvViT"></d-cite>. 

Despite the complementary nature of these architectures, they break the fidelity of the transformer and make for difficult analysis. Therefore, there exists a gap in the traditional transformer architecture to perform in small-data regimes, particularly in vision. Motivated by this shortcoming, we aim to investigate and improve the current ViT paradigm to narrow the gap between CNNs and ViTs on small-data. In particular, we examine novel initialization schemes, removal of component parts in our transformer block, and new-learnable parameters which can lead to better performance, image throughput, and stable training on small-scale datasets.

<div class="col-sm text-center">
  <a name="figure-3"></a>
  
  <img src="{{ 'assets/img/2023-11-08-diaz-proposal/transformer.svg' | relative_url }}" class="img-fluid rounded z-depth-1" style="width: 300px;">
  
  <div class="caption">
    Figure 3: Standard transformer encoder block. Encoder can be stacked for x amount of layers.
  </div>
</div>

### Transformer Block
To serve as a basis of comparison, we will examine the stanford transformer block seen in [Figure 3](#figure-3). The block is identical to <d-cite key="AttentionIsAllYouNeed"></d-cite> with the exception of using layer normalizations before the multi-headed attention (MHA) and multi-level perceptron (MLP) blocks as opposed to after. In practice, this placement has been shown to be more stable and increase performance<d-cite key ="Liu2020"></d-cite>. With the exception of this modification, the block has seen little improvements over the years testifying to its robustness. However, recent trends in theory hints towards ways we could break this notion – all while enjoying increased performance.

Before we delve into these advances and their implications, consider the following transformer block information flow:

$$
\displaylines{
\text{Attention} = \text{A}(X) = \text{Softmax}\Biggl(\frac{XW_{Q}W_{K}^{T}X^{T}}{\sqrt{k}}\Biggl) 
\\ \\ 
\text{A}(X) \in \mathbb{R}^{T\times T}}
$$

which is shortly followed by:

$$
\displaylines{
  \text{S}(X) = \text{A}(X)W_{V}W_{O}
\\ \\
\text{S}(X) \in \mathbb{R}^{T\times d}
}
$$

and:

$$
\text{Output} = \text{MLP}(\text{S}(X))= \text{Linear}(\text{GELU}(\text{Linear}(\text{S}(X))))
$$


where:

* Embedded input sequence: $$X \in \mathbb{R}^{T \times d}$$
* Linear queury and key layers: $$W_{Q},W_{K} \in \mathbb{R}^{d \times k}$$
* Linear value and projection layers: $$W_{V}, W_{O} \in \mathbb{R}^{d \times d}$$
* MLP Linear layers: $$\text{Linear} \in \mathbb{R}^{d \times d}$$
* $$T = $$ \# of tokens, $$d = $$ embedding dimension, $$k = \frac{d}{H}$$, $$H = $$ \# of attention heads

The flow of information mirrors the transformer block in [Figure 3](#figure-3). Readers unfamiliar with transformer intricacies such as MHA and MLPs are encouraged to read<d-cite key="AttentionIsAllYouNeed"></d-cite>.

Recently, there have been many proposals on how the transformer block can be further modified to increase data throughput and eliminate “redundant” or “useless” parts that do not have any significant contribute to the tranformer's modeling capabilities. For example, <d-cite key = "2302.05442"></d-cite>, used a parallel MHA and MLP incorporated into a large-scale ViT for stable and efficient training. Throughout this blog, we will focus on the ideas overviewed and proposed by <d-cite key = "He2023"></d-cite> as they present intriguing results and a synthesis on the current state of this research topic. The interested reader is encouraged to study their paper for a more extensive understanding of the ideas.


<div class="col-sm text-center">
  <a name="figure-4"></a> 
  {% include figure.html path="assets/img/2023-11-08-diaz-proposal/simplified_block.png" class="img-fluid rounded z-depth-1" %}
  <div class="caption"> 
    Figure 4: Comparison between trasnformer architectures. <em>Left</em>: Standard block as shown in Figure 3. <em>Bottom Right</em>: Parallel block proposed in. <em>Top Right</em>: Newly proposed encoder.
  </div>
</div>

The overaching theme of <d-cite key = "He2023"></d-cite> was to take the standard trasnformer block and evaluate the necessity of each component. In doing so, they removed each component part and studied its effects on performance. Understandably, blindly removing components will lead to unstable training and ill-performance (i.e. if one were to remove the skip connnections, they would encounter vanishing gradients as seen [Figure 14](#figure-14)). However, <d-cite key = "He2023"></d-cite> took the approach of removal combined with recovery. For example, when the authors removed skip connections, they required a modification to the self-attention matrix of the form:

$$
\text{A}(X) \leftarrow (\alpha\text{I} + \beta \text{A}(X))
$$

where $$\alpha$$ and $$\beta$$ are learnable scalars and intialized to $$1$$ and $$0$$, respectively, and $$\text{I} \in \mathbb{R}^{T \times T}$$ is the identity matrix. This modification intiailizes the self-attention matrix providing a pathway towards training stability. They further entertained a more complicated scheme with a third parameter, but we only consider the two parameter version for simplicity. By this iterative removal and recovery process, the authors converged towards the final transformer block seen in [Figure 4](#figure-4). The most shocking aspect of this proposed block is the removal of the $$W_{V}$$ and $$W_O$$ layers. They arrived to this justification by initialializing $$W_{V}$$ and $$W_{O}$$ to the identity with separate, learnable scalars and training a model. Over the course of training, the scalar ratios converged towards zero<d-footnote>This is a slight simplification. Look at Section 4.2 and Figures 4 and 20 in He et. al 2023 for a more detailed explanation.</d-footnote>. Due to the heavy cost and speed these linear layers present, removal of them decreases parameters counts and enables more data throughput. A concise PyTorch interpretation of the new block can be seen below:  

```python
import torch
import torch.nn as nn

class ShapedAttention(nn.Module):
    def __init__(self, width: int, n_hidden: int, num_heads: int):
        super().__init__()
        # Determining if hidden dimension of attention layer is divisible by number of heads
        assert width % num_heads == 0, "Width and number of heads are not divisble."
        
        # Setting vars
        self.head_dim   = n_hidden // num_heads
        self.num_heads  = num_heads
        # Creating Linear Layers
        self.W_K = nn.Linear(width, self.head_dim)
        self.W_Q = nn.Linear(width, self.head_dim)
        # Learnable Scalars: alpha_init and beta_init are up to user
        self.alpha = nn.Parameter(alpha_init)
        self.beta = nn.Parameter(beta_init)
        # Softmax
        self.softmax = nn.Softmax(dim = -1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input:
        # x: shape (B x T x dim)
        # Outputs:
        # attn_output: shape (B x T x width)
        attn_output = None
        # Compute keys and queries
        k = self.W_K(x)
        q = self.W_Q(x)
        # Scaled dot-product
        attn_scores = torch.bmm(q, k.transpose(1,2)) / (self.head_dim**-0.5)
        attn_scores = self.softmax(attn_scores)
        # Shaped attention
        B, T, _ = x.shape
        output = self.alpha*torch.eye(T, device = x.device) + self.beta * attn_scores

        return output
```

The performance of the final transformer block (referred to as SAS-P) demonstrated powerful results. In the [Figure](#figure-5), the simplified transformer matches the standard block in cross-entropy loss even when taken through a long runtime. Additionally, Figure 6 in <d-cite key = "He2023"></d-cite> demonstrates the model scales better with depth which is of paramount importance in modern neural network.

<div class="col-sm text-center">
  <a name="figure-5"></a> 
  {% include figure.html path="assets/img/2023-11-08-diaz-proposal/fig5.png" class="img-fluid rounded z-depth-1" %}
  <div class="caption"> 
    Figure 5: Training speed experiment. Figure 6. in Ref. 15. Pre-LN is the standard transformer block. SAS-P is the block. It is shown with and without an initial layer normalization.
  </div>
</div>


## Translation to Vision: Experimentation and Analysis
The results shown in <d-cite key = "He2023"></d-cite> show promise the transformer can be improved. Motivated by vision applications, we seek to implement such infrastructure, with slight modifications, and determine if it improves performance in small datasets.

### Vanilla vs. Simplified Comparison
For evaluation, we compare the simplified transformer to a vanilla ViT. The vanilla ViT's transformer block is identical to the formulation presented earlier. We use Conv2D patch embedding with a random initial positional embedding. For the simplified setup, we initialize $$\alpha = \beta = 0.5$$ and do not use a centering matrix – although it has been shown to improve ViT performance<d-cite key = "2306.01610"></d-cite>. We use one Layer Normalization just prior to the transformer encoder. $$\text{Width} = 96$$ is kept uniform throughout the model. The dataset is CIFAR-10 with a $$\text{batch size} = 256$$. Data augmentations were random horizontal and vertical flips with 15º random rotations. Optimizer is AdamW with $$\text{lr} = 0.003$$ and $$\text{weight decay} = 0.01$$. We employ a cosine learning rate scheduler to maintain consistency with ViT literature, although <d-cite key = "He2023"></d-cite> empirically showed a linear learning rate is slightly advantegeous<d-footnote>Figure 11 in He et. al 2023.</d-footnote>. We ran our model for $$\text{epochs} = 50$$ with $$\text{runs} = 3$$ to evalute run-to-run stability. A condensed version of the experiment choices can be seen in Table 1. The results can be seen in [Figure 6](#figure-6) and Table 2<d-footnote>To avoid clutter, only the training accuracies vs epochs are shown. Loss curves showed similar relationships.</d-footnote>.

| Table 1.  Experiment 1: ViT Model Settings  |   |
| ------------- |----|
| # of channels | 3  |
| Image size    | 32 |
| Patch size    | 4  |
| Width         | 96 |
| # of heads    | 4  |
| # of layers   | 8  |

<div class="l-page" style="display: flex; flex-direction: column; align-items: center; margin-bottom: 0px;">
  <a name="figure-6"></a> 
  <iframe src="{{ 'assets/html/2023-11-08-diaz-proposal/exp1_w96.html' | relative_url }}" frameborder='0' scrolling='no' height="500px" width="70%"></iframe>
  <div class="caption" style="margin-top: 10px; text-align: center;"> 
    Figure 6. Comparison between vanilla and simplified transformers. Width = 96. Layers/Depth = 8.
  </div>
</div>

| Table 2. Experiment 2: Results         |    Vanilla       | Simplified  |  $$\Delta$$   |
| ------------- |:-------------:| -----:|----:|
|  Parameters     | 358186 | 209210 | -41.59%      |
|  Avg. epoch time (s) | 12.954       |    11.305 |      -12.73% |

Experiment 1 showed the training evaluation trajectory is nearly identicable between the two models although the simplified outperforms by small margin. Although the subtle difference, it is noteworthy to mention the simplified version achieved mirroring performance with less parameters and higher image throughput. The similarity of the curves hints the removal of the skip connections, layer normalizations, and value/projection layers were merited, begging the question whether these components held our modeling power back.


This experimentation shows the similar nature of each model, but does not translate well to wider modern neural networks. In Experiment 2, we expanded to $$\text{width} = 128 $$ to determine if there is any emergent behaviour as the network becomes wider. We replicate everything in Experiment 1 and solely modify the width. The settings are restated in Table 3. The results for Experiment 2 can be seen in [Figure 7](#figure-7) and Table 4 below.


| Table 3   |  Experiment 2: ViT Model Settings  |
| ------------- |----|
| # of channels | 3  |
| Image size    | 32 |
| Patch size    | 4  |
| Width         | 128 |
| # of heads    | 4  |
| # of layers   | 8  |

<div class="l-page" style="display: flex; flex-direction: column; align-items: center; margin-bottom: 0px;">
  <a name="figure-7"></a> 
  <iframe src="{{ 'assets/html/2023-11-08-diaz-proposal/exp2_w128.html' | relative_url }}" frameborder='0' scrolling='no' height="500px" width="70%"></iframe>
  <div class="caption" style="margin-top: 10px; text-align: center;"> 
    Figure 7. Comparison between vanilla and simplified transformers. Width = 128. Layers/Depth = 8.
  </div>
</div>


| Table 4. Experiment 2: Results         |    Vanilla       | Simplified  |  $$\Delta$$   |
| ------------- |:-------------:| -----:|----:|
|  Parameters     | 629130 | 364954 | -41.99%      |
|  Avg. epoch time (s) | 13.093      |    11.735 |      -10.37% |

The narrative is different for Experiment 2. The simplified version outperforms the vanilla version by a considerable margin. An adequate explanation for this discrepancy in vision tasks merits further exploration. However, considering the proposed unnecessary nature of the value and projection matrices, we can hypothesize they interfere with the modeling capability as more parameters are introduced. 

Due to the sheer difference in outcomes between the models, we question how the models are attending towards various inputs to gain a better understanding of what is happening under the hood. To probe this curiosity, we trained the models with identical setting in Experiment 2, but modified the $$\text{depth} = \text{layers} = 12$$. This model setup will be covered 
in more detail in future paragraphs. We inputted CIFAR-10 to each model and visualized a side-by-side comparison of attention maps for five input images. An interactive figure is seen [Figure 8](#figure-8).

<div class="l-page" style="display: flex; flex-direction: column; align-items: center; margin-bottom: 0px;">
  <div style="display: flex; flex-direction: column; align-items: center;">
    <a name="figure-8"></a> 
    <iframe src="{{ 'assets/html/2023-11-08-diaz-proposal/attention_maps.html' | relative_url }}" frameborder='0' scrolling='no' height="600px" width="70%"></iframe>
    <div class="caption" style="margin-top: 10px; text-align: center;"> 
      Figure 8. Comparison between vanilla and simplified attention maps. Width = 128. Layers/Depth = 12. Interpolation method: "nearest".
    </div>
  </div>
</div>
There is a noticeable contrast in the attention maps. For the simplified model, the attention maps seem to place weight in a deliberation manner, localizing the attention towards prominent features in the input image. On the other hand, the vanilla model is choatic in its attention allocation. It is noteworthy that the vanilla model does place attention towards areas of interest, but also attends towards irrelevant information perhaps compromising its judgement at the time of classification. It can thus be reasoned the simplified model can better decipher which features are relevant demonstrating, even in low data regimes, the representational quality is increased.

While we have so far investigated width, it will be informative to understand how depth impacts the performance of the simplified version. In <d-cite key = "He2023"></d-cite>, they employ signal propagation theory, which is most prominent in deeper networks. Therefore, we suspect as we increase the depth of our models, the simplified version will outperform the vanilla version by a larger margin. Here, we set $$\text{layers} = 12$$ and maintain $$\text{width}=128$$. The training accuracies and experiment results are seen in [Figure 9](#figure-9) and Table 5.

<div class="l-page" style="display: flex; flex-direction: column; align-items: center; margin-bottom: 0px;">
  <a name="figure-9"></a> 
  <iframe src="{{ 'assets/html/2023-11-08-diaz-proposal/exp3_w128_l12.html' | relative_url }}" frameborder='0' scrolling='no' height="500px" width="70%"></iframe>
  <div class="caption" style="margin-top: 10px; text-align: center;"> 
    Figure 9. Comparison between vanilla and simplified transformers. Width = 128. Layers/Depth = 12.
  </div>
</div>

| Table 5. Experiment 3: Results         |    Vanilla       | Simplified  |  $$\Delta$$   |
| ------------- |:-------------:| -----:|----:|
|  Parameters     | 927370 | 531106 | -42.72%      |
|  Avg. epoch time (s) | 17.527     |    15.723 |      -10.29% |

Again, the simplified model outperforms the vanilla model by a large margin. Although we have focused on performance in the past, we discern an interesting trend when we scaled the depth: the simplified version seemed to be more consistent from run-to-run (recall $$\text{runs} = 5$$). This leads us to believe that as we continue to scale the depth, the simplified version will be more stable. Future experimentation will be necessary to corroborate this claim.






### Initialization Schemes
We have seen the impact simplification can have on the performance of the transformer performance and self-attention. However, the used initializatons of $$\alpha$$ and $$\beta$$ in Experiments 1, 2, and 3, was based on equal weighting between the initial attention matrix and the identity matrix. In <d-cite key = "He2023"></d-cite>, they employ a full weighting of the identity matrix and zero'd out the attention matrix at initialization. Here, we aim to determine the effect of different initialization values. Recall $$\alpha = \beta = 0.5$$ in Experiments 1, 2, 3. Now, we investigate two more initializaton schemes: $$\alpha = 1.0$$ and $$\beta = 0.0$$ and vice-versa. We replicate the protocol used in Experiment 2 and only modify these learnable scalar at initializaton and set $$\text{runs} = 1$$. The results are shown in [Figure 10](#figure-10). Interestingly, the initialization scheme proposed by <d-cite key = "He2023"></d-cite>, does *not* outperform the equal weighting or inverse weighting scheme. Understandably, it does poorly at initialization, but never recovers. The equal weighting and inverse weighting approaches show nearly identical performance often trading off superior performance from epoch-to-epoch.



<div class="l-page" style="display: flex; flex-direction: column; align-items: center; margin-bottom: 0px;">
  <a name="figure-10"></a> 
  <iframe src="{{ 'assets/html/2023-11-08-diaz-proposal/exp4_init_new.html' | relative_url }}" frameborder='0' scrolling='no' height="500px" width="80%"></iframe>
  <div class="caption" style="margin-top: 10px; text-align: center;"> 
    Figure 10. Various Initialization Schemes.
  </div>
</div>

This lead us to believe the initializaton scheme could be improved. There has been some work on initializing vanilla ViTs<d-cite key = "Trockman2023"></d-cite> to gain performance. In <d-cite key = "Trockman2023"></d-cite>, a prominent diagonal was observed for the $$W_{q}W_{k}^{T}$$ layers in ViT's pre-trained on large datasets, which have been shown to outperform  CNNs. The figure shown in the paper can be seen in [Figure 10](#figure-10). This motivated the authors to provide a novel initialization scheme where the $$W_{Q}$$ and $$W_{K}$$ matrices are initialized in a way to encourage diagonal prominence in the forward pass. However, our findings contradicted this scheme, as our diagonal-dominant initialization scheme $$\alpha = 1$$ and $$\beta = 0$$ did not out perform the inverse or the equal weighting. This is likely due to the fact we have learnable parameters and do not initialize our $$W_{Q}$$ and $$W_{K}$$'s directly, but rather the attention matrix post-softmax. However, it is important to realize that the learnable parameters still encourage diagonal prominence regardless of intialization. Although<d-cite key = "Trockman2023"></d-cite> used this initialization scheme to increase performance in small ViT's trained from scratch, which encourages tokens to attend toward to themselves through the depth of the network, they did not take into consideration how the diagnolization varys from layer-to-layer. Seen in [Figure 10](#figure-10), we can see the prominence of the diagnoal elements fades as we go deeper into the network. Observing this behaviour, we hypothesize the reason the initialization scheme of $$\alpha = 1$$ and $$\beta = 0$$ underperformed was not due to the initialization itself, but how it was applied to each layer. In other words, when we initialized $$\alpha = 1$$ and $$\beta = 0$$, we encouraged this token self-attentive nature throughout the depth of the network, when we should be encouraging it in the opening layers and tapering it off as we approach the end of the model. 

To give more evidence to this hypothesis, we experimented with the following dynamic initialization scheme:

$$
\displaylines{
\alpha_i = \frac{1}{i}, \beta_i = 1 - \frac{1}{i} \\
 \text{ where } i \in [1, 2, ..., L] \text{ and } L = \text{# of layers}
}
$$

The results from this initialization scheme compared to the uniform initializations can be seen in [Figure 12](#figure-12) The results show that the dynamic scheme outperform the results perhaps indicating the representation quality is connected toward encouraging self-token connection in the lower layers, while allowing for token's to intermingle in higher layers. We further experiment with the inverse dynamic where we switch the $$\alpha$$ and $$\beta$$ values. The results in [Figure 13](#figure-13) show the dynamic approach is stronger during training then the inverse dynamic approach.

<div class="col-sm text-center">
  <a name="figure-11"></a> 
  {% include figure.html path="assets/img/2023-11-08-diaz-proposal/diagonal_vit_tiny.png" class="img-fluid rounded z-depth-1" %}
  <div class="caption"> 
    Figure 11: Diagonal prominence in a pre-trained ViT Tiny. Layers 1-11 (Left-to-Right). Heads 1-3 (Top-to-Bottom). Extracted from Figure 1 of <a href="https://arxiv.org/abs/2305.09828">Mimetic Initialization of Self-Attention Layers</a>.
  </div>
</div>

<div class="l-page" style="display: flex; flex-direction: column; align-items: center; margin-bottom: 0px;">
  <a name="figure-12"></a> 
  <iframe src="{{ 'assets/html/2023-11-08-diaz-proposal/exp5_init_dynamic.html' | relative_url }}" frameborder='0' scrolling='no' height="500px" width="80%%"></iframe>
  <div class="caption" style="margin-top: 10px; text-align: center;"> 
    Figure 12. Experiment 5: Dynamic vs. Uniform Initializations.
  </div>
</div>

<div class="l-page" style="display: flex; flex-direction: column; align-items: center; margin-bottom: 10px;">
  <a name="figure-13"></a> 
  <iframe src="{{ 'assets/html/2023-11-08-diaz-proposal/exp6_init_inverse.html' | relative_url }}" frameborder='0' scrolling='no' height="500px" width="80%"></iframe>
  <div class="caption" style="margin-top: 10px; text-align: center;"> 
    Figure 13. Experiment 6: Dynamic vs. Inverse Dynamic Initializations.
  </div>
</div>

## Conclusion and Limitations
Through this blog post we have overviewed the simplification of our known transformer block and novel initialization schemes. We took the problem of small-scale training of ViT's and looked to address it leveraging such ideas. Through a series of experiments and thoughtful schemes, we generated an informed and sophisticated approach to tackle such a problem. In the end, we generated a method that outperformed a tradtional ViT in small scales. We explored ways of scaling the ViT in width and depth and probed how the new model distributed attention. Our comparisons were intentionally simple and effective in addressing the underlying task and illustrating the models potential. Although the results presented showed promise, extensive validation needs to be performed in the future. It will be interesting to see how this new transformer block and intialization scheme can be further utilized in computer vision. For example, a logical next route to entertain is to compare convergence rates in larger scale ViT on datasets such as ImageNet-21k to see if the modeling advantage persists.

There are a few limitations in this study. For one, only one dataset was used. Using other datasets such as CIFAR-100 or SVHN would provide more insight into this methodology. Secondly, there is a need for more comprehensive evaluation and ablation studies to determine the true nature of the simplified transformer and initialization schemes. Third, a comparison to a smaller scale CNNs is needed to gauge where this method comparatively sits in modeling power. 


<div class="l-page" style="display: flex; flex-direction: column; align-items: center; margin-bottom: 10px;">
  <a name="figure-14"></a> 
  <iframe src="{{ 'assets/html/2023-11-08-diaz-proposal/exp0.html' | relative_url }}" frameborder='0' scrolling='no' height="500px" width="80%"></iframe>
  <div class="caption" style="margin-top: 10px; text-align: center;"> 
    Figure 14. Experiment 0: Removal of skip connections in traditional ViT.
  </div>
</div>


