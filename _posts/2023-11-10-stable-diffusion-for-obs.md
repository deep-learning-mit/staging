---
layout: distill
title: Stable Diffusion for Oracle Bone Script
description: The project aims to train a ControlNet for Stable Diffusion on the condition of rendering traditional Chinese characters from oracle bone script samples.
date: 2023-12-12
htmlwidgets: true

authors:
  - name: Jenny Moralejo
    url: 
    affiliations:
      name: MIT

# must be the exact same name as your blogpost
bibliography: 2023-11-10-stable-diffusion-for-obs.bib  

# Add a table of contents to your post.
#   - make sure that TOC names match the actual section names
#     for hyperlinks within the post to work correctly.
toc:
  - name: Introduction
  - name: Related Work
  - name: Methodology and Experiments
    subsections:
    - name: Dataset Construction
    - name: Dataset Preprocessing
    - name: Training
    - name: Experiments
  - name: Experiment Analysis
    subsections:
    - name: Locked Stable Diffusion
    - name: Unlocked Stable Diffusion
  - name: Discussion
    subsections:
    - name: Limitations and Future Work
    - name: Conclusion
  - name: Project Resources
  - name: Notes + Troubleshooting

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

## Introduction
{% include figure.html path="assets/img/2023-11-10-stable-diffusion-for-obs/oracle_bone_example.jpg" class="img-fluid rounded" %}
<div class="caption">
Fig. 1: Inscribed tortoise carapace (“oracle bone”), Anyang period, late Shang dynasty, c. 1300–1050 B.C.E.
</div>

Oracle bone script (甲骨文) is an ancient form of Chinese characters which were engraved on bones (mostly of turtle and oxen) (Fig. 1). These bones were used to record historical events and communicate with ancestors and deities to predict disaster and bestow fortune. Many of these characters have been matched to their modern day depictions however the meaning of many still remain unknown. Currently scientists have only deciphered 1,000 of the over 4,000 identified characters <d-cite key="divinity2023"></d-cite>. 

Many early day Chinese characters originate from a pictographic base. The characters in oracle bone script often, but not always, share structural similarities to their modern day counterparts and one is able to trace development of the character throughout time (Fig. 2).

{% include figure.html path="assets/img/2023-11-10-stable-diffusion-for-obs/horse_evolution.png" class="img-fluid rounded" %}
<div class="caption">
Fig 2. Evolution of the "horse" character from oracle bone script to traditional and simplified Chinese.
</div>

The deciphering of these characters holds immense historical and cultural value, giving researchers insight to the lives and beliefs of society at the time, and is an active field of research today. 

The goal of this project is to train a ControlNet for stable diffusion to transform images of the oracle bone script characters to their modern day traditional Chinese counterparts.

## Related Work

Many different approaches have been used to classify oracle bone script characters including using deep learning to compute the similarity between a rubbing and characters in the oracle bone script font library <d-cite key="zhangObs2020"></d-cite>, leveraging pretrained networks such as YOLO and MobileNet <d-cite key="dlObs2022"></d-cite>, building out and training specialized convolutional neural networks <d-cite key="HWOBC"></d-cite>, and using hierarchical representations<d-cite key="Hierarchical"></d-cite>. 
These methods were very successful at classifying and grouping oracle bone script characters. Accuracies on the HWOBC dataset were as high as 97.64% using the DCNN Melnyk-Net<d-cite key="HWOBC"></d-cite>. 

While these methods are able to group both deciphered and undeciphered oracle bone script characters, these groupings are not informative of what the deciphered oracle bone script character may be. There are very few approaches that have applied machine learning and generative artificial intelligence to decipher these characters. Traditionally deciphering these characters relies on the knowledge and experience of professional historians applying rules on OBS evolution. 

New methods and frameworks have been suggested to help aid the deciphering of these inscriptions including a new case based system proposed by Zhang et al. which given a query of a character returns a set of similar characters and other data to help aid in the deciphering process. This framework utilizes an autoencoder to find similarities between adjacent writing systems tracking the evolution of the character from OBS to its modern day counterpart. However the accuracies of translating from OBS to Bronze Epigraphs only almost reached 50% when considering the top 100 character categories. Additionally this work only translated directly from OBS to Bronze Epigraphs to Chu State Characters and this work is incomplete when it comes to the deciphering of unknown characters <d-cite key="Decipher2021"></d-cite>. 

The approach of Wang et al. attempted to trace the evolution of these characters using the VGG16 network showing the inheritance and similarity between characters; however, it also did not generate the deciphering of unknown characters<d-cite key="Evolution2022"></d-cite>.

Using machine learning and generative techniques to decipher OBS characters is an emerging field and current work deciphering relies on access to Bronze Epigraphs and Seal Script. While these intermediate writing systems provide valuable information and structure on character evolution, they are difficult to link together and I attempt to bypass direct knowledge of these intermediate mappings to translate directly between OBS and traditional Chinese. 

Recently many developments have been in generative AI space, especially in the development of Stable Diffusion to gradually denoise images. As ControlNet is able to leverage the existing large diffusion model U-Net with layers trained on billions of images, it is able to learn a diverse set of conditional controls<d-cite key="ControlNet"></d-cite>. I hope to leverage this and train a ControlNet that can control the generation of modern day Chinese characters given their OBS counterpart.

While it is unknown if many of these characters even have modern day counterparts, by training the model to perform well on known oracle bone script and modern day traditional Chinese character pairs I hope to generate a "best guess" on what the deciphered character could be.

## Methodology and Experiments
### Dataset Construction 
There are a variety of publicly available OBS datasets online. However, many datasets contain images of oracle bone rubbings that are often very noisy and even cropped and fragmented. The images in the collected datasets may not even be of the same size (Fig. 3). This variety, while useful for many classification tasks and more pertinent real world applications would be a larger challenge when training a ControlNet. 

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/2023-11-10-stable-diffusion-for-obs/obc306-1.bmp" class="img-fluid"  %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/2023-11-10-stable-diffusion-for-obs/obc306-2.bmp" class="img-fluid" %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/2023-11-10-stable-diffusion-for-obs/obc306-3.bmp" class="img-fluid" %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/2023-11-10-stable-diffusion-for-obs/obc306-4.bmp" class="img-fluid" %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/2023-11-10-stable-diffusion-for-obs/obc306-5.bmp" class="img-fluid" %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/2023-11-10-stable-diffusion-for-obs/obc306-6.bmp" class="img-fluid" %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/2023-11-10-stable-diffusion-for-obs/obc306-7.bmp" class="img-fluid" %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/2023-11-10-stable-diffusion-for-obs/obc306-8.bmp" class="img-fluid" %}
    </div>
</div>
<div class="caption">
Fig. 3: Oracle bone script rubbings belonging to the same character class from the OBC306 dataset
</div>

The dataset I chose is the [HWBOC database](https://jgw.aynu.edu.cn/ajaxpage/home2.0/DataOBC/detail.html?sysid=22) created by Bang Li et al.<d-cite key="HWOBC"></d-cite>. This was the largest publicly available dataset I could find with clean images. The dataset contains 83245 images of oracle bone script handwriting samples divided into 3881 classes. Of these 3881 characters 1457 characters have been deciphered into their modern day equivalent. The dataset images are 400x400 and the characters are relatively centered around the same size (Fig. 4). 

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/2023-11-10-stable-diffusion-for-obs/hwobc-1.png" class="img-fluid"  %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/2023-11-10-stable-diffusion-for-obs/hwobc-2.png" class="img-fluid" %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/2023-11-10-stable-diffusion-for-obs/hwobc-3.png" class="img-fluid" %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/2023-11-10-stable-diffusion-for-obs/hwobc-4.png" class="img-fluid" %}
    </div>
</div>
<div class="caption">
Fig. 4: HWOBC dataset images for the character disaster (災/灾)
</div>

ControlNet relies on having "source" and "target" image pairs to train on. Using the mapping between the characters to their deciphered counterparts, I generated traditional Chinese character images for every deciphered OBS character. The file mapping the known OBS characters to their traditional counterparts can be found in Project Resources. For simplification, each image class has the same target image (a many to one mapping between the "source" images and the "target") (Fig. 5).  The reason for this is that there are many "correct" ways of etching a character in oracle bone script that would map to the same Chinese character. Many OBS characters can be flipped in multiple directions and still represent the same character. This is not a characteristic of modern day Chinese. The images are generated using the KaiTi Chinese font library. An additional important simplification I made was that sometimes a class of OBC characters has multiple modern day counterparts -- I only generated a target image for the first given counterpart in the list. 
<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/2023-11-10-stable-diffusion-for-obs/617CA_5.png" class="img-fluid"  %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/2023-11-10-stable-diffusion-for-obs/617CA_7.png" class="img-fluid" %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/2023-11-10-stable-diffusion-for-obs/617CA_17.png" class="img-fluid" %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/2023-11-10-stable-diffusion-for-obs/617CA_FTZ.png" class="img-fluid" %}
    </div>
</div>
<div class="caption">
Fig. 5: Left OBS "sources" corresponding to the rightmost traditional Chinese "target" of the Yang character (陽/阳)
</div>

For OBS characters which are separated into multiple characters in their modern day translation, I made the decision to generate the modern day counterpart by stacking the characters vertically. However, different composite characters are sometimes created through horizontal stacking. This is a limitation that can be improved upon in future iterations. 

Finally in order to guide the training process, a prompt is required. I was hoping to leverage some existing knowledge of what a Chinese character is within the U-Net and because the tasks are all the same the prompt was "traditional Chinese character"

### Dataset Preprocessing 
Due to stable diffusion models currently needing to take an input with dimensions divisible by 64, I resized all of the images to 384x384. The dataset was further sampled from with an 80-20 split to create a validation and a training set (from the set of deciphered classes with a "target" image). Additionally, as some characters are unable to be rendered with font libraries and different encodings, the classes 6131A, 60EBA, and 6100F were removed from the dataset. As the images were already binary and without noise, no further preprocessing steps were taken. 

The script to create the JSON file mapping source to target images with a prompt as well as target image generation can be found in Project Resources.

In total, the model was trained on 31337 pairs of unique "source" images and "target" images (of which there were 1454).

### Training
As ControlNet freezes the Stable Diffusion U-Net and creates trainable copies of certain blocks, the copies and "zero convolution" blocks can receive a condition to integrate into the main model (Fig. 6). 

{% include figure.html path="assets/img/2023-11-10-stable-diffusion-for-obs/controlNet.png" class="img-fluid rounded"  %}
<div class="caption">
Fig. 6: Visualization of the ControlNet setup
</div>

I chose to control the standard Stable Diffusion model [SD1.5](https://huggingface.co/runwayml/stable-diffusion-v1-5/tree/main).
In order to train the model, I modified the scripts found in the [main ControlNet Github](https://github.com/lllyasviel/ControlNet/tree/main) to create the dataloader object, attach the ControlNet to the pretrained existing Stable Diffusion model, and train the model.

The model was run on a Vast.ai computing instance with an A100PCIE machine. 

### Experiments
In total I trained a ControlNet on this dataset twice: once with the Stable Diffusion model layers locked and again with the Stable Diffusion layers unlocked. 
#### Locked Stable Diffusion
By default, the Stable Diffusion layers in the controlled model are locked. This allows the model to maintain the pretrained parameters learned from the larger dataset. This option is usually beneficial when training a ControlNet on smaller datasets with a larger array of prompts and conditions. 
Due to the smaller size of the constructed dataset, I wanted to see the performance when the layers are locked. When training with the locked layers, I set the learning rate to 1e-5 and used a batch size of 4. In total, this model was trained for 2 epochs.
#### Unlocked Stable Diffusion
However, a benefit of unlocking the Stable Diffusion layers is that increased performance has been found for more specific problems. By unlocking the original Stable Diffusion model layers, both the model and the ControlNet are being simultaneously trained (Fig. 7). Due to the performance of the previous model, I also increased the batch size from 4 to 8 to decrease training time. Additionally I decreased the learning rate to 2e-6 to be more careful when learning as to not degrade the capability of the original Stable Diffusion model. This model was run for a total of 5 epochs,

{% include figure.html path="assets/img/2023-11-10-stable-diffusion-for-obs/unlocked-SD.png" class="img-fluid rounded"%}

<div class="caption">
Fig. 7: Visualization of the ControlNet setup with Stable Diffusion layers unlocked
</div>


## Experiment Analysis
### Locked Stable Diffusion
Sudden convergence for ControlNet is usually observed around 7000 steps or after the first epoch in this case. This model was only trained for 2 epochs and 21370 steps. Sudden convergence never occurred for the model. The model improved quite quickly to looking like real Chinese characters to the untrained eye with noticeable improvements in structure as well as writing style between the steps 0 to 300 to 3000 (Fig. 8). Noticeably we see that without the ControlNet in step 0 the results are quite far from the desired. However even in later steps, none of the generated characters are real.

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/2023-11-10-stable-diffusion-for-obs/locked-SD-0.png" class="img-fluid"  %}
        <div class="caption">
          Step 0
        </div>
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/2023-11-10-stable-diffusion-for-obs/locked-SD-300.png" class="img-fluid"  %}
        <div class="caption">
          Step 300
        </div>
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/2023-11-10-stable-diffusion-for-obs/locked-SD-3000.png" class="img-fluid"  %}
        <div class="caption">
          Step 3000
        </div>
    </div>
</div>
<div class="caption">
Fig. 8: Training sampling checkpoint results, from top to bottom: source OBS image, generated image, target Chinese image
</div>

In the first couple thousand steps of training as demonstrated in the Fig. 8, there aren't many matching structural similarities between the characters, this began to change around step 10000. 
<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/2023-11-10-stable-diffusion-for-obs/locked-SD-10835.png" class="img-fluid"  %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/2023-11-10-stable-diffusion-for-obs/locked-SD-10835-annotated.png" class="img-fluid"  %}
    </div>
</div>
<div class="caption">
Fig. 9: Training sampling checkpoint results from step 10835 radical mappings and similarities circled in red, top to bottom: source OBS image, generated image, target Chinese image
</div>

The results from Fig. 9 are really exciting to see as finally it looks as if the model is starting to learn the structural radical mappings between OBS and traditional Chinese. These mappings are not correct or complete but it is a step up from results in previous steps and similarities between characters are easier to see. Unfortunately, model improvement slowed down and did not appear to be improving past step 20000 (Fig. 10).

{% include figure.html path="assets/img/2023-11-10-stable-diffusion-for-obs/locked-SD-21070.png" class="img-fluid"  %}
<div class="caption">
Fig. 10: Training sampling checkpoint results from step 21070, top to bottom: source OBS image, generated image, target Chinese image
</div>

As seen in the above figures many of the generated characters would have some sort of greyscale background. My prompt did not specify the background to be white and was instead just "traditional Chinese character". I suspect the model was using the varying grayscale backgrounds to "cheat" a little when it came to how closely the reconstructed image matched the target. Another important observation is that the model often generated images with many more strokes than the target image. I suspect this is because of the distribution of target images being slightly more skewed to more complicated characters than simple ones. 

Overall the model did show capability of learning the objective, however further tweaks are needed to increase its performance.

### Unlocked Stable Diffusion
This model was trained by unlocking the underlying Stable Diffusion model layers the ControlNet was attached onto. In total the model trained for 20790 steps with the modified parameters specified above. The performance between the models at first appears quite similar. We observe that without the ControlNet the generated image is not as expected with great improvements in steps 300 and 3000 (Fig. 11). 
<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/2023-11-10-stable-diffusion-for-obs/unlocked-SD-0.png" class="img-fluid"  %}
        <div class="caption">
          Step 0
        </div>
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/2023-11-10-stable-diffusion-for-obs/unlocked-SD-300.png" class="img-fluid"  %}
        <div class="caption">
          Step 300
        </div>
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/2023-11-10-stable-diffusion-for-obs/unlocked-SD-3000.png" class="img-fluid"  %}
        <div class="caption">
          Step 3000
        </div>
    </div>
</div>
<div class="caption">
Fig. 11: Training sampling checkpoint results with unlocked SD layers, from top to bottom: source OBS image, generated image, target Chinese image
</div>

The difference between the models becomes apparent during further timesteps. While this result is not very common, in further timesteps there is actually occurences of the side radical in the generated character matching the target. While there were relationships between the character radicals in the locked version (Fig. 9), they were never in the exact form as circled in Figure 12. 

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/2023-11-10-stable-diffusion-for-obs/unlocked-SD-12054.png" class="img-fluid"  %}
        <div class="caption">
          Step 12054
        </div>
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/2023-11-10-stable-diffusion-for-obs/unlocked-SD-17472.png" class="img-fluid"  %}
        <div class="caption">
          Step 17472
        </div>
    </div>
</div>
<div class="caption">
Fig. 12: Training sampling checkpoint results with unlocked SD layers, from top to bottom: source OBS image, generated image, target Chinese image
</div>

Both models seem to preserve the general structure of the character, be it left to right or top to bottom. However the characters generated past ~ step 10000 of the unlocked model look much more realistic than the locked model. This is due to the fact that actual modern day Chinese radicals comprise the generated characters (although the characters generated are still not real). Additionally, the images generated by unlocking Stable Diffusion layers do not have the same problem as the locked layer with the grayscale background.

Overall the model did show capability of learning the objective and there was improved performance from unlocking the layers and increasing batchsize, however further tweaks are needed to increase its performance. In particular, I think the model would benefit from being run for even longer or increasing the batch size.

## Discussion
### Limitations and Future Work
Due to computational and financial limitations, I was unable to run either of the models for longer than 20000 steps and was unable to use a batch size of larger than 8. While sudden convergence is commonly observed around 7000 steps, this is not always the case and I think the model would benefit from being trained for longer or if feasible with a larger batch size so that training does not take as long. 

Additionally, a current limitation is that I did not augment the data at all. Currently all of the characters are relatively centered and are of the same size, in the future it would be of interest to uncenter the characters (ie: have them be placed in the upper left corner etc.). I suspect this could potentially enforce more structural and radical adherence compared to current performance of the model. Furthermore, analysis should be done into the distribution of the target data to better understand why the results of the ControlNet are the way they are and to potentially remove outliers from the dataset. Specifically, the distribution with regards to character complexity which could be quantified roughly by measuring the filled pixels or with more sophistication by counting the number of strokes. 

I am also curious to see how training a ControlNet on this data would perform with a different prompt. Perhaps specifying a white background would cause the model to converge faster. Additionally the knowledge of what a "traditional Chinese character" is could be different than its understanding of "繁体字" (traditional Chinese character in Chinese). I would love to further explore how changing the prompt could improve results or speed up time to convergence. 

Finally, due to the hierarchical structural nature of Chinese characters exploring a way to somehow further reinforce these representations within the ControlNet architecture would also be beneficial. 

### Conclusion
Both models able to form structural level connections between oracle bone script and traditional Chinese. While the generated characters were not actual Chinese it was also able to emulate the form of actual Chinese using many existing radicals. The model with unlocked stable diffusion layers outperformed the locked model qualitatively generating more realistic characters.

Although the trained models are not able to translate with confidence between oracle bone script and traditional Chinese, this method shows promise as a tool to further aid the research into deciphering oracle bone script.

***

## Project Resources
1. [HWBOC database](https://jgw.aynu.edu.cn/ajaxpage/home2.0/DataOBC/detail.html?sysid=22)
2. [Dataset Creation Resources](https://github.com/jmortan/OBS-ControlNet)
5. [ControlNet Github](https://github.com/lllyasviel/ControlNet/tree/main) 
6. [SD1.5 Model](https://huggingface.co/runwayml/stable-diffusion-v1-5/tree/main)

***

## Notes + Troubleshooting
1. To unzip the file with HWOBC dataset - use software that supports the Chinese encoding (ie: Chinese WinRAR)
2. To train ControlNet image size must be divisible by 64 
3. When creating an instance of a machine dependency errors arise if the version of Cuda and PyTorch specified in the image does not align with specifications for the ControlNet training