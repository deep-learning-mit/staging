---
layout: distill
title: Zero-Shot Machine-Generated Image Detection using Sinks of Gradient Flows
description: "How can we detect fake images online? A novel approach of characterizing the behavior of a diffusion model's learned score vectors."
date: 2023-11-08
htmlwidgets: true

# Anonymize when submitting
# authors:
#   - name: Anonymous

authors:
  - name: Marvin Li
    url: ""
    affiliations:
      name: Harvard
  - name: Jason Wang
    url: ""
    affiliations:
      name: Harvard

# must be the exact same name as your blogpost
bibliography: 2023-11-08-detect-image.bib  

# Add a table of contents to your post.
#   - make sure that TOC names match the actual section names
#     for hyperlinks within the post to work correctly.
toc:
  - name: Abstract
  - name: Introduction
  - name: Related Work
  - name: Methods
  - name: Experiments
  - name: Discussion

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

### Abstract

Detecting AI-generated content has become increasingly critical as deepfakes become more prevalent. We discover and implement algorithms to distinguish machine-generated and real images without the need for labeled training data. We study the problem of identifying photorealistic images using diffusion models. In comparison to the existing literature, we discover detection techniques that do not require training, based on the intuition that machine-generated images should have higher likelihoods than their neighbors. We consider two metrics: the divergence of the score function around a queried image and the reconstruction error from the reverse diffusion process from little added noise. We also compare these methods to ResNets trained to identify fake images from existing literature. Although the previous methods outperform out methods in terms of our accuracy metrics, the gap between our zero-shot methods and these ResNet methods noticeably declines when different image transformations are applied. We hope that our research will spark further innovation into robust and efficient image detection algorithms. 

### Introduction

As AI-generated images become ever more widespread, garnering virality for how realistic they have become, we are increasingly concerned with the potential for misuse. A deluge of machine-generated fake images could spread misinformation and harmful content on social media. From relatively innocuous pictures of [Pope Francis](https://www.nytimes.com/2023/04/08/technology/ai-photos-pope-francis.html) wearing an AI-generated image puffer coat to dangerous [disinformation campaigns](https://www.politico.eu/article/ai-photography-machine-learning-technology-disinformation-midjourney-dall-e3-stable-diffusion/) powered by diffusion models, we live in a new era of media that we cannot trust. The European Union has passed [legislation](https://www.nytimes.com/2023/12/08/technology/eu-ai-act-regulation.html) that, among other regulations, requires AI-generated content to be explicitly marked so. The enforcement of such legislation and similar-minded policies, however, remains unclear. Consequently, a growing body of research has sought to develop techniques to distinguish between the real and the synthetic.

The rise of models capable of generating photorealistic content makes the detection problem difficult. While there are still numerous nontrivial challenges with current models from their inability to depict text and render tiny details humans are innately sensitive to such as eyes and hands, the pace of the technology is moving in a way that makes relying on these flaws short-sighted and dangerous. Another potential complication is that advanced photo editing techniques such as [Adobe Firefly](https://www.adobe.com/products/firefly.html) have capabilities such as generative inpainting that make it such that an image could contain both real and invented content. Even simple data augmentations like crops, rotations, color jitters, and horizontal flipping can make the input look vastly different to a detection model. Furthermore, the majority of popular image generation tools are text-conditional, and we cannot expect to recover the text prompt, not to mention the model that generated the image. This makes transferable, zero-shot techniques of paramount importance.

In this paper, we propose two techniques for detecting images from diffusion models (see Figure [1](#fig-methods-illustrated)). Diffusion models <d-cite key="sohl2015deep"></d-cite> have been one of the most successful architectures for image generation, inspired by thermodynamic principles. Diffusion models learn a score function (gradient of log likelihood) that 'undoes' noise from the image. In effect, these models learn a gradient field that points to the real-world data manifold.<d-cite key="batzolis2022your"></d-cite> We leverage the intuition that the greater the deviation the diffusion model's machine-generated images are from the real world data, the greater the difference of the neighborhood gradient field. In particular, we believe that machine-generated images are more likely to live in a 'sink' of the gradient field as the diffusion model 'flows' images down the gradient field. We thus propose the *divergence of a diffusion model's score function* as a promising zero-shot statistic for whether an image is generated by the diffusion model.

In addition, another metric for the 'sink' property of the gradient field at the image of concern is how far the image moves after a small displacement and flow along the gradient field. This has a nice interpretation in diffusion models as the *reconstruction error* for running the reverse process over just a small timestep on just a slightly perturbed image.

*<a name="fig-methods-illustrated">Figure 1:</a> The Divergence and Reconstruction Error Hypothesis: Images on the generated data manifold <span style="color: red">(red)</span> have negative divergence and small reconstruction error, while images on the real data manifold <span style="color: green">(green)</span> have zero divergence and large reconstruction error.*
{% include figure.html path="assets/img/2023-11-08-detect-image/methods-illustrated.png" class="img-fluid" %}

Our overarching research question is thus summarized as, can we use the properties of a diffusion model's tacit vector field to build an effective zero-shot machine-generated image detector, specifically looking at *divergence* and *reconstruction error*?

The main contributions of our paper are:

1. Proposing two methods inspired by sinks of gradient flows: *divergence* and *reconstruction error*.

2. Conducting a wide battery of experiments on the performance of these methods in a variety of augmentation settings.

### Related Work

Previous literature has considered several different methods for image detection. Sha et al. 2022 <d-cite key="sha2022fake"></d-cite> trained machine learning classifiers to detect fake images using high-level image and text embeddings. They, however, do not consider the local information around image embeddings, and require existing datasets of known image-generated and non-image-generated examples to train their classifier. Corvi et al. 2023 <d-cite key="corvi2023detection"></d-cite> identified "forensic traces" in machine-generated image residuals for this task. Again, their method requires many data samples, and requires separate training on diffusion models and GANs. 

We are inspired by ideas from DetectGPT,<d-cite key="mitchell2023detectgpt"></d-cite> a recent work which addressed the same problem of detecting AI-generated content, but in the setting of large language models. For a given piece of text, DetectGPT perturbs the original text and computes the difference in log-likelihood between the perturbed text and the original text:

$$\mathrm{DetectGPT}(x,p_{\theta},q)\triangleq\log p_{\theta}(x)-\mathbb{E}_{\tilde{x}\sim q(\cdot|x)}\log p_{\theta}(\tilde{x})$$

where $p_\theta$ is the language model and $q$ is the distribution of perturbations. If the difference in log-likelihood is large, then the attack claims that the original text is more likely to be generated by a language model.

There are several critical differences between language models and diffusion models. With text, one can directly compute the log likelihood of a given piece of text, even with only blackbox access, i.e., no visibility to the model's parameters. In contrast, for diffusion models, it is intractable to directly compute the probability distribution over images because diffusion models only learn the score. Moreover, the most commonly used diffusion models, e.g. DALL-E 3, apply the diffusion process to a latent embedding space rather than the pixel space. To address the latter concern, we plan on applying the encoder to the image to obtain an approximation of the embedding that was passed into the decoder. And to address the former, instead of approximating the probability curvature around a given point like DetectGPT, we formulate a statistic characterizing whether the gradient field/score is a sink, i.e., the gradients around a machine-generated image point to the machine-generated image. This captures the idea of a local maximum in probability space, similar to the DetectGPT framework. 

It would be remiss to not mention Zhang et al. 2023,<d-cite key="zhang2023watermarks"></d-cite> who argued that watermarking, a strictly easier task than machine-generated image detection, is likely impossible. They claim that an adversary who can perturb a generated image of text without too much degradation and has blackbox access to the watermarking scheme can conduct a random-walk on reasonable outputs until the watermark is degraded. However, their analysis was mainly theoretical and lacked specific experiments with diffusion models. It remains to be seen whether their assumptions still hold for image generation, and whether more concrete watermarking schemes may afford some level of protection against less sophisticated adversaries or the unintentional use of machine-generated images. 

### Methods

**Dataset.** To conduct our research, we needed datasets of known real and fake images. We used MSCOCO <d-cite key="lin2014microsoft"></d-cite>, a dataset of 330K non-machine generated images and captions of common real-world objects which was also used by Corvi et al. 2023.<d-cite key="corvi2023detection"></d-cite> Initially, we planned to use DiffusionDB <d-cite key="wang2022diffusiondb"></d-cite> for our fake images, a dataset of 14M (prompt, image) pairs generated by the open-source Stable Diffusion Version 1 model scraped from the StableDiffusion discord. However, we realized that many of the images in DiffusionDB are not meant to be realistic. Instead, we iterated through the captions of MSCOCO and used Stable Diffusion V1.4  to generate a matching machine-generated image for that caption, as in Corvi et al. 2023.<d-cite key="corvi2023detection"></d-cite>

**Baseline.** We used the model and code from Corvi et al. 2023 <d-cite key="corvi2023detection"></d-cite> to identify images generated by Stable Diffusion as our trained baseline. Their model is a ResNet18 image-only detector trained on the training split of the MSCOCO dataset and images also generated by prompts from StableDiffusion.

**Detection Algorithms.** For out attacks, we compute the divergence of the diffusion model's score field around the image (negative divergence indicates a sink). We can estimate this via a finite-differencing approach: given a diffusion model $s_\theta(x)$ which predicts the score $\nabla_x\log p_\theta(x)$, we have that

$$\mathrm{div}(s_\theta,x)= \sum_{i=1}^d \frac{s_\theta(x+he_i)_i-s_\theta(x-he_i)_i}{2h}$$

for small $h$ and orthogonal basis $\{e_i\}_{i=1}^d$.
However, images are high-dimensional, and even their latent space has $\approx10,000$ dimensions, which means that fully computing this sum could be computationally expensive. In this paper, we sample a fraction of the dimensions for each queried image.  

Another way to capture the intuition that machine-generated images are have higher likelihoods than their neighbors is by noising the latent to some timestep $t$, and then comparing the distance of the denoised image to the diffusion model to the original image. That is, given a diffusion model $f_\theta$ which takes a noised image and outputs an unnoised image (abstracting away noise schedulers, etc. for clarity),

$$\mathrm{ReconstructionError}(f_{\theta},x)\triangleq \mathbb{E}_{\tilde{x}\sim \mathcal{N}(x,\epsilon)}||x-f_{\theta}(\tilde{x})||_2^2$$

for small $\epsilon$. The intuition is that if an image and thus more likely, then the denoising process is more likely to send noisy images to that particular image. 

**Comparison.** For each model, we use the AUC-ROC curve and the true positive rate (TPR) at low false positive rate (FPR) as metrics. The latter notion of accuracy is borrowed from the membership inference attack setting in Carlini et al. 2021.<d-cite key="carlinifpr"></d-cite> As they argue, this metric quantifies our confidence that a point identified as fake is actually fake. In important settings like filtering fake images on social media platforms, this is especially important as there may be asymmetric consequences for accidentally flagging an image as fake compared to missing a fake image. We also provide a data visualization tool for the images our method identifies. In the real world, we can expect that the images we want to test will be distorted, either by random cropping, reflections, rotations, or compression. We will apply image augmentations over both fake and real image datasets and report the same metrics over these augmentations. 

### Experiments

We run all experiments over a common set of 500 images from the test set of [MSCOCO](https://huggingface.co/datasets/nlphuji/mscoco_2014_5k_test_image_text_retrieval) and the corresponding 500 images generated by Stable Diffusion V1.4 with the same prompt using HuggingFace's default arguments.

For our Divergence method, we randomly sample $d=10$ dimensions to compute the divergence over and set $h=0.1$. For our Reconstruction method, we compute an average distance over 10 reconstructed images per original image and use add/remove noise equivalent to 1 time-step.

For each method, we evaluate the performance on no augmentation, random $256\times 256$ crop (corresponding to about a quarter of the image for generated images), grayscale, random horizontal flip with probably $0.5$, random rotation between $[-30^\circ,30^\circ]$, and random color jitter of: brightness from $[0.75,1.25]$, contrast from $[0.75,1.25]$, saturation from $[0.75,1.25]$, and hue from $[-0.1,0.1]$.


*<a name="table-results">Table 1:</a> Divergence, Reconstruction, and ResNet Detection AUC and True Positive Rate at 0.1 False Positive Rate.*
<table>
    <tr>
        <th>AUC / TPR$_{0.1}$</th>
        <th colspan="3" style="text-align: center">Method</th>
    </tr>
    <tr>
        <th>Augmentation</th>
        <th>Divergence</th>
        <th>Reconstruction</th>
        <th>ResNet</th>
    </tr>
    <tr>
        <th>No Aug.</th>
        <td>0.4535 / 0.078</td>
        <td>0.7310 / 0.000</td>
        <td>1.000 / 1.000</td>
    </tr>
    <tr>
        <th>Crop</th>
        <td>0.4862 / 0.092</td>
        <td>0.4879 / 0.064</td>
        <td>1.000 / 1.000</td>
    </tr>
    <tr>
        <th>Gray.</th>
        <td>0.4394 / 0.056</td>
        <td>0.7193 / 0.000</td>
        <td>1.000 / 1.000</td>
    </tr>
    <tr>
        <th>H. Flip</th>
        <td>0.4555 / 0.084</td>
        <td>0.7305 / 0.000</td>
        <td>1.000 / 1.000</td>
    </tr>
    <tr>
        <th>Rotate</th>
        <td>0.4698 / 0.062</td>
        <td>0.6937 / 0.000</td>
        <td>0.9952 / 0.984</td>
    </tr>
    <tr>
        <th>Color Jitter</th>
        <td>0.4647 / 0.082</td>
        <td>0.7219 / 0.000</td>
        <td>1.000 / 1.000</td>
    </tr>
</table>

*<a name="fig-roc-auc">Figure 2:</a> AUC-ROC Curves in No Augmentation Setting.*
<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        <p>(a) Divergence</p>
        {% include figure.html path="assets/img/2023-11-08-detect-image/method=Divergence_n_points=500_n_samples=10_noise_amount=0.1_num_inference_steps=25_seed=229_bs=1_roc.png" class="img-fluid"  %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        <p>(b) Reconstruction</p>
        {% include figure.html path="assets/img/2023-11-08-detect-image/method=Reconstruction_n_points=500_n_samples=10_noise_amount=1.0_num_inference_steps=25_seed=229_bs=1_roc.png" class="img-fluid" %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        <p>(c) ResNet</p>
        {% include figure.html path="assets/img/2023-11-08-detect-image/method=Resnet_n_points=1000_seed=229_bs=1_roc.png" class="img-fluid" %}
    </div>
</div>

*<a name="fig-hists">Figure 3:</a> Histograms of Computed Statistics in No Augmentation Setting.*
<div class="l-body">
  <p>(a) Divergence</p>
  <iframe src="{{ 'assets/html/2023-11-08-detect-image/method=Divergence_n_points=500_n_samples=10_noise_amount=0.1_num_inference_steps=25_seed=229_bs=1.html' | relative_url }}" frameborder='0' scrolling='no' height="600px" width="100%"></iframe>
</div>
<div class="l-body">
  <p>(b) Reconstruction</p>
  <iframe src="{{ 'assets/html/2023-11-08-detect-image/method=Reconstruction_n_points=500_n_samples=10_noise_amount=1.0_num_inference_steps=25_seed=229_bs=1.html' | relative_url }}" frameborder='0' scrolling='no' height="600px" width="100%"></iframe>
</div>
<div class="l-body">
  <p>(c) ResNet</p>
  <iframe src="{{ 'assets/html/2023-11-08-detect-image/method=Reconstruction_n_points=500_n_samples=10_noise_amount=1.0_num_inference_steps=25_seed=229_bs=1.html' | relative_url }}" frameborder='0' scrolling='no' height="600px" width="100%"></iframe>
</div>

**Trained Baseline.** The trained baseline does extraordinarily well at the MSCOCO vs. Stable Diffusion detection task. It achieves $1.0$ AUC (perfect accuracy) across all augmentation settings except for rotation for which it gets an almost perfect AUC of $0.9952$. This high performance matches Corvi et al. 2023's findings,<d-cite key="corvi2023detection"></d-cite> stemming from the fact that the ResNet was trained on the MSCOCO distribution and Latent Diffusion generated images are similar to Stable Diffusion generated images. In their paper, the performance noticeably drops to around $0.7$-$0.8$ AUC for other image generation models.

**Divergence.** Divergence does extremely poorly, with AUCs just slightly below 0.5, indicating that in fact generated images have greater divergence than real images---the opposite of our intuition, but this may also be noise as these values are essentially equivalent to random guessing. We suspect that this is largely due to our low choice of $d$, meaning that we cannot get a representative enough sample of the dimensions to get an accurate estimate of the true divergence. We may have also chosen $h$ too large, as we have no idea of the scale of any manifold structure that may be induced by the gradient field.

**Reconstruction Error.** Reconstruction error, on the other hand, boasts impressive AUCs of around $0.7$. The shape of the curve is particularly strange, and with the additional observation that the AUC when the random cropping is applied goes back to $0.5$ AUC, indicated to us that the image size may be the differentiating factor here.  MSCOCO images are often non-square and smaller than the $512\times 512$ constant size of the generated images. As the Frobenius norm does not scale with image size, we hypothesize that using the spectral norm and dividing by the square root of the dimension would instead give us a more faithful comparison, akin to the random crop results. However, data visualization of the examples does not show a clear correlation between image size and reconstruction error, so it appears that this detection algorithm has decent AUC but poor TPR at low FPR, and is vulnerable to specifically cropping augmentations.

<a href="http://jsonw0.pythonanywhere.com/">**Detection Visualizations.**</a> We developed a dashboard visualizaiton that enables us to look more closely at images and their associated detection statistics. Some examples we can pick out that seem to make sense include Figure 4, where the real image is captioned as a CGI fake image, and predictably gets a low statistic as deemed by Reconstruction Error (the generated image, ironically, gets a higher statistic denoting more real).

*<a name="fig-methods-illustrated">Figure 4:</a> An Example Image of a CGI "Real" Image Getting Detected as Fake.*
{% include figure.html path="assets/img/2023-11-08-detect-image/cgi-example.png" class="img-fluid" %}

However, from a visual inspection of images, we cannot identify a clear relationship between image content or quality of generated images that holds generally. We make our dashboard public and interactive; a demo can be seen below:

<div class="l-screen">
  <iframe src="http://jsonw0.pythonanywhere.com/" frameborder='0' scrolling='yes' height="1200px" width="100%"></iframe>
</div>


### Discussion

Throughout our experiments, the divergence-based detector performs much worse than the other detectors. Because the latent space has a very high dimension, the divergence detector may require sampling from many more dimensions than is practical for an image detector in order to obtain good estimates of the divergence. Further research should try to scale this method to see if it obtains better results. Mitchell 2023 et al. <d-cite key="mitchell2023detectgpt"></d-cite> justifies the validity of their machine-generated as a Hutchinson trace estimator of the divergence of the log probabilities; however, the poor performance of the divergence detector imply that estimating the trace is not helpful for image detection and that other model properties may instead be at play for this method's effectiveness. In contrast, the noising/denoising detector implicitly incorporates information from all dimensions, which may explain its better performance. The model from Corvi et al. 2023 <d-cite key="corvi2023detection"></d-cite> outperforms our methods under all augmentations, achieving a perfect AUC on images without data augmentations. This is consistent with what was reported in their manuscript. However, this is not an unbiased estimate of the trained classifier's performance, because they also used MSCOCO data to train and test their classifier. We were limited to this experimental setup by data availability and previous literature. Future work should comapre the zero-shot and trained detectors on completely out-of-sample data and with different generation models.

Although at face-value our detectors perform worse than the pre-trained model in our experiments, our project still introduces some interesting ideas for machine-generated image detection that are of interest to the broader community and worth further exploring. First, the techniques we explored parallel zero-shot machine-generated image detection methods for text.<d-cite key="mitchell2023detectgpt"></d-cite> The fact that in both settings, perturbing the inputs and computing the curvature of the log probabilities are potent signals for machine-generated detection implies that these features may be an indelible mark of machine-generated models across all modalities. Second, image detection algorithms trained on data may be fundamentally vulnerable to adversarial modifications. Because there exists non-robust features that are predictive of the output in training data,<d-cite key="ilyas2019adversarial"></d-cite> adversaries, who realistically may have access to the image detection algorithm over many trials, can craft subtle background noise that circumvents image-detection algorithms. Our methods, which consist of only a few parameters, are not prone to adversarial attacks unlike trained models. Third, this work highlights the use of other features besides the image as features for image detection, e.g. score function and noising/denoising the image. Future work may build on the ideas behind these features to improve trained image detectors.