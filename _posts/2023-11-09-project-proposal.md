---
layout: distill
title: Scale-Aware Multiple-Instance Vision-Language Contrastive Learning 
description: We present a novel approach for the diagnosis of renal pathologies from electron microscopy (EM) images utilizing deep learning. Our method leverages CLIP, a self-supervised vision-language model, to bridge the gap between unstructured textual diagnostic reports and EM images. By introducing a learnable scale embedding, our model becomes scale-aware, capturing disease features at various resolutions. Additionally, we propose a multiple-instance image encoder to learn a single patient-level embedding from a set of multiple images. We train our model on a dataset comprising 600,000 EM images across 15,000 patients, along with their diagnostic reports. Using a held-out test set, we evaluate our model on diverse tasks including zero-shot diagnosis, retrieval, and feature probing.

date: 2023-11-09
htmlwidgets: true

# Anonymize when submitting
# authors:
#   - name: Anonymous

authors:
  - name: Andrew Zhang
    url: "https://www.linkedin.com/in/azh22/"
    affiliations:
      name: HST, MIT
  - name: Luca Weishaupt
    url: "https://www.linkedin.com/in/luca-weishaupt/"
    affiliations:
      name: HST, MIT


# must be the exact same name as your blogpost
bibliography: 2023-11-09-project-proposal.bib

# Add a table of contents to your post.
#   - make sure that TOC names match the actual section names
#     for hyperlinks within the post to work correctly.
toc:
  - name: Proposal
#   - name: Images and Figures
#     subsections:
#     - name: Interactive Figures
#   - name: Citations
#   - name: Footnotes
#   - name: Code Blocks
#   - name: Layouts
#   - name: Other Typography?

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

## Proposal

Many vision tasks are analogous to “finding a needle in a haystack”, where only a small portion of the image is relevant. This is especially true in the field of pathology, where only a few cells in a biopsy image may contain disease features. Because these images are so large, it is often advantageous to examine them at multiple scales <d-cite key='chenScalingVisionTransformers2022'></d-cite>. In September of 2023, it was shown that in addition to positional embeddings, using scale embeddings for image analysis tasks with deep learning can be incredibly beneficial for analyzing satellite imagery <d-cite key="reedScaleMAEScaleAwareMasked2023"></d-cite>. We see a clear parallel between analyzing vast amounts of satellite imagery and analyzing large medical images in digital pathology to make a diagnosis. 

In the field of renal pathology, electron microscopy (EM) is a crucial imaging modality for diagnosing diseases such as amyloidosis and thin membrane disease, amongst many others. A pathologist has to analyze up to 90 EM images per patient, at vastly different scales (ranging from 2 to 100 nanometers per pixel), to make a diagnosis. While deep learning methods have been proposed for automatically classifying a disease from single images in a supervised fashion <d-cite key="hackingDeepLearningClassification2021, zhangDeepLearningbasedMultimodel2023"></d-cite>, in the field of medical imaging labels suitable for supervised training often do not exist. For example renal pathologists generate a full report in unstructured text, addressing the EM findings in the context of the patient’s clinical background. Therefore, in order to make a scalable AI system which can take advantage of the vast amounts of unstructured medical data, self-supervised methods are necessary. We propose 
1. to use an unsupervised vision-language model to create an expressive and scalable shared embedding space between textual descriptions for diagnoses and EM images 
2. to learn a patient-level single embedding corresponding to multiple images, in the way that a pathologist would use multiple images to make a diagnosis and 
3. to add a learnable scale embedding after extracting their features, in order to make the image encoder scale-aware. 

Through nefarious means, we have obtained a dataset containing 600,000 renal EM images corresponding to 15,000 patients, along with a written diagnostic report for each patient. We will adapt the CLIP architecture for multiple-instance scale-aware contrastive learning between the images for each patient and their diagnostic report. Following self-supervised CLIP pretraining, we will evaluate the model on the following tasks: Zeroshot diagnosis on a held-out test set, retrieval at the patient-level and image-level, and linear probing of the learned image features. We will compare the performance of our model to a baseline model which does not use scale embeddings.

Deliverables:
- A baseline multiple-instance CLIP model without scale embeddings
- A multiple-instance CLIP model with scale embeddings
- AUC and balanced accuracy on the zero-shot diagnosis task and linear probing task
- Recall@K on the retrieval tasks

{% include figure.html path="assets/img/2023-11-09-project-proposal/Fig1.png" class="img-fluid" %}
