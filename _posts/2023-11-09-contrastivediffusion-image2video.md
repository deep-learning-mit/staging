---
layout: distill
title: Robust Image to Video Generation Using Contrastive Diffusion Over Latents
description: Image-to-video (I2V) may be the next frontier of generative deep learning capabilities, but current models struggle with robustness, largely due to the implicit, rather than explicit, representation learning objective during traditional diffusion model training. Hence, we propose a new technique where a pre-trained contrastive model is used to train a diffusion model with a custom contrastive loss function to operate within a learned structured latent space for I2V problems, yielding, in theory, more structurally sound videos without loss of contextual information.
date: 2023-11-09
htmlwidgets: true

# Anonymize when submitting
# authors:
#   - name: Anonymous

authors:
  - name: Rishab Parthasarathy
    affiliations:
      name: MIT CSAIL
  - name: Theo Jiang
    affiliations:
      name: MIT CSAIL

# must be the exact same name as your blogpost
bibliography: 2023-11-09-contrastivediffusion-image2video.bib  

# Add a table of contents to your post.
#   - make sure that TOC names match the actual section names
#     for hyperlinks within the post to work correctly.
toc:
  - name: Introduction and Motivation
  - name: Related Work
  - name: Proposed Project Outline
  - name: Evaluation
    subsections:
      - name: Generation Quality
      - name: Use of Contrastive Latent Space
  - name: Implementation/Deliverables

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

## Introduction and Motivation

With recent advances in computer vision and generative AI, we all have observed the various feats that diffusive models have achieved in conditional image generation. These models have demonstrated unparalleled ability in creativity, fidelity, and relevance when generating images from text prompts. Given this explosive success of diffusion for the task of image generation, the idea of applying the same concepts to conditional video generation seems like a logical follow-up. Yet, the field still lacks robust and compelling methods for conditional video generation with diffusion models. This raises the question: why might this be? Or perhaps a follow-up: what makes videos so hard in comparison to images?

In an attempt to address our first question, if we take a brief dive into previous literature, we will find that the issue is not a lack of effort. Ho et al. <d-cite key ="ho2022video"></d-cite>, Zhang et al. <d-cite key="2023i2vgenxl"></d-cite>, and Chen et al. <d-cite key = "chen2023videocrafter1"></d-cite>, all explore this idea, yet the results from these methods are not nearly as exciting as the results we see in images. But why is this? 

Perhaps the answer lies in the solution to our second question. One of the most obvious complexities that videos have over images is also perhaps one of the most difficult: the temporal dependence between frames. But why is this relationship so hard for diffusion models? Following the work of Zhu et al. <d-cite key = "zhu2022discrete"></d-cite>, we hypothesize that this is because the implicit learning of diffusive steps between images in a video is too complex of a problem for a diffusion model; relying on the model to learn the implicit relationship between representations of video frames is infeasible from a training and convergence standpoint. If we can instead learn diffusive steps over a more regularized learned latent space, the optimization problem can be greatly simplified and the diffusion model will in theory be more robust.


## Related Work

Taking a step back to examine the current state of research, we find that current image-to-video frameworks typically still use a traditional diffusion architecture, going straight from text and image representations to an output image. However, this naive approach struggles with serious issues like frame clipping and loss of contextual information, which is expected since noise-based sampling can easily throw off the output of individual frames.

Hence, Ho et al. in 2022 proposed the first solution, supplementing conditional sampling for generation with an adjusted denoising model that directly forces image latents to be more similar to the corresponding text latents <d-cite key ="ho2022video"></d-cite>. While this achieved improved results over the straightforward diffusion approach, this often forces the model to stick too closely to the text latent, resulting in incoherent videos. 

To solve this issue, two recent approaches from Chen et al. and Zhang et al. have proposed methods to augment the video diffusion models themselves. Chen et al. uses the image encodings from CLIP-like language embeddings in an encoder-decoder language model, feeding the CLIP encodings at each step into a cross-attention layer that generates attention scores with the current video generation <d-cite key = "chen2023videocrafter1"></d-cite>. In doing so, additional coherence between frames is achieved. On the other hand, Zhang et al. use multiple encoders, with CLIP and VQ-GAN concatenated before two stages of diffusion model training, which they claim provides the hierarchical learning required to learn the temporal processing <d-cite key="2023i2vgenxl"></d-cite>. However, both these models are extremely data-heavy and still suffer from hallucination and frame skipping.

To remedy these issues in diffusion models, Ouyang et al. and Zhu et al. posit that the implicit representation learning objective in diffusion models is the primary cause of the slow convergence and hallucination issues. Specifically, diffusion models do not directly compare their output to their input, as in contrastive models, instead performing a variational approximation of the negative log-likelihood loss over the full Markov chain. Instead, Ouyang and Zhu propose to train the diffusion model to output a structured latent in the latent space of a contrastive model like a VQ-VAE, which then reconstructs the output image <d-cite key = "zhu2022discrete"></d-cite> <d-cite key = "ouyang2023improving"></d-cite>. In doing so, a contrastive term can be added to the loss of the diffusion model, maximizing the mutual information between the structured (output) latent and input latent, leading to stronger correlations between input and output, and hence improved convergence. Hence, this approach seems to have potential in fixing the hallucination and coherence issues in video diffusion models, without the need for added complexity.


## Proposed Project Outline

Thus, we propose a novel method for conditional video generation (generating videos given a starting frame and text description) by utilizing an autoencoder framework and contrastive loss to train a regularized latent space in which a diffusion model can operate. Following the line of thought introduced above, we hypothesize that under such a formulation, the diffusion model is much more robust to temporal inconsistency, because of the regularity in the latent space. For example, if we imagine a highly regularized latent space, we will find all logical next frames for a given anchor frame clustered very closely around the anchor in this latent space. Therefore, any step the diffusion model takes would produce valid subsequent frames; it suffices simply for the model to learn which direction to go given the conditioned text prompt. 

With this in mind, we detail the construction of the model by describing its components as follows:
1. An encoder for image data is used to map a given video frame into our latent space
1. An encoder for text data is used to map a given video description into our latent space
1. A diffusion-based model operates within the latent space, diffusing between different vectors within this latent space.
1. A decoder is used to generate images from vectors in this latent space.

The training process of such a model will involve the optimization of a diffusion/contrastive loss based on a given pair of adjacent video frames, as well as the corresponding text description for that video. We define a training step to involve the following:
1. Both video frames and the text description are encoded into our latent space.
1. One iteration of our diffusive model is run by diffusing from the latent vector corresponding to our earlier frame conditioned on our text prompt latent to obtain a new latent vector.
1. This new latent vector after cross-attention is passed through the decoder to obtain our predicted subsequent frame.
1. We then optimize our model according to the contrastive diffusion model loss presented by <d-cite key = "zhu2022discrete"></d-cite> with a key alteration: we replace their contrastive loss with our contrastive loss, which contains two terms:
    1. a term that aims to push our two adjacent video frames closer together in our latent space and
    2. a term that aims to push video frames closer to the text description in our latent space.

During inference, we generate a video through the following process:
1. An initial frame and the text description are encoded into our latent space
1. We run an arbitrary number of diffusive steps, generating a latent at each step.
1. We decode the latent at each time step to obtain our video frame at that time step; stringing these frames together produces our video.

From a more theoretical perspective, this method essentially aims to restrict the diffusion model’s flexibility to paths within a highly regularized, lower dimensional latent space, as opposed to the entire space of images that classical diffusion-based approaches can diffuse over. Such a restriction makes it much harder for the diffusion model to produce non-sensible output; the development of such a method would therefore enable the robust generation of highly temporally consistent and thus smooth videos. We also imagine the value of producing such a latent space itself. An interesting exercise, for example, is taking an arbitrary continuous path along vectors within a perfectly regular latent space to obtain sensible videos at arbitrary framerates.


## Evaluation

There are two axes along which we wish to evaluate our model: quality of generation, and quality of the contrastive latent space.

### Generation Quality

To measure generation quality, we follow the approach presented by Ho et al., evaluating famous metrics like the FID, FVD, and IS scores. For all of these metrics, we expect to evaluate them throughout the video from beginning to end, with the level of preservation of metric values throughout a video indicating consistent video quality. Similarly, we will compare our models to those of similar size using the same metrics to evaluate whether adding the contrastive loss term truly improves generation quality. These metrics will be supplemented with qualitative human analyses, where we will score the videos on a variety of axes including coherence and relevance to the prompt.

### Use of Contrastive Latent Space

Given that the diffusion model now maps to a much smaller latent space when compared to the whole space of output images, we believe that the diffusion output should have interpretable representations in the latent space. Hence, we will begin by exploring the latents generated by different text prompts, clustering them around the image source encodings to evaluate if the contrastive loss has truly clustered appropriately. On top of that, we plan to visualize the trajectories of videos for both the training set and our generations, to evaluate our theory of continuous trajectory evolution in the latent space.

## Implementation/Deliverables

The implementation of such a method can be greatly simplified through the use of an existing codebase. We plan on using the contrastive diffusion model [GitHub repository](https://github.com/L-YeZhu/CDCD/tree/main) for the implementation of <d-cite key = "zhu2022discrete"></d-cite> with a few key modifications:
- We use a pre-trained contrastive model as our starting point (such as an image encoder/decoder from CLIP) <d-cite key = "Radford2021LearningTV"></d-cite>
- The diffusion model is trained to predict the next frame of a video conditioned on a given text description of the video and the current frame of the video as above.
- Our contrastive loss is used as described above.
- Inference is modified to generate a video as described above.

Data for this project requires video/text description pairs. There are a few datasets consisting of such data, including the [MSR-VTT dataset](https://www.kaggle.com/datasets/vishnutheepb/msrvtt), which is human-annotated, and the [InternVid dataset](https://github.com/OpenGVLab/InternVideo/tree/main/Data/InternVid), which is annotated by LLMs. 

The project should be feasible to complete within the remaining time in the semester, with a rough timeline of deliverables as follows:
- **Implementation** of our method by applying the specified modifications to the existing codebase should take around 1-2 weeks.
- **Training** of the models on cloud computing resources should take <1 week.
- **Evaluation and benchmarking** along with data visualization should take 1 week, even with the potential need for retraining our models.
- **Blog writing** should take <1 week and can be completed in parallel with evaluation and benchmarking.