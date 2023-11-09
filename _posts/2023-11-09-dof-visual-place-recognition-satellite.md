---
layout: distill
title: 6-DOF estimation through visual place recognition
description: A neural Visual Place Recognition solution is proposed which could help an agent with a downward-facing camera (such as a drone) to geolocate based on prior satellite imagery of terrain. The neural encoder infers extrinsic camera parameters from camera images, enabling estimation of 6 degrees of freedom (6-DOF), namely 3-space position and orientation. By encoding priors about satellite imagery in a neural network, the need for the agent to carry a satellite imagery dataset onboard is avoided.
date: 2023-11-09
htmlwidgets: true

# Anonymize when submitting
# authors:
#   - name: Anonymous

authors:
  - name: Andrew Feldman (Kerberos: abf149)
    url: "https://andrew-feldman.com/"
    affiliations:
      name: MIT

# must be the exact same name as your blogpost
bibliography: 2023-11-09-dof-visual-place-recognition-satellite.bib  

# Add a table of contents to your post.
#   - make sure that TOC names match the actual section names
#     for hyperlinks within the post to work correctly.
toc:
  - name: Introduction
  - name: Background
#  - name: Images and Figures
#    subsections:
#    - name: Interactive Figures
  - name: Proposed solution
    subsections:
    - name: Image-to-extrinsics encoder architecture
    - name: Data sources for offline training
    - name: Training and evaluation
      subsections:
      - name: Data pipeline
      - name: Training
      - name: Hyperparameters
      - name: Evaluation

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

# Introduction

The goal of this project is to demonstrate how a drone or other platform with a downward-facing camera could perform approximate geolocation through visual place recognition, using a neural scene representation of existing satellite imagery.

Visual place recognition<d-cite key="Schubert_2023"></d-cite> refers to the ability of an agent to recognize a location which it has not previously seen, by exploiting a system for cross-referencing live camera footage against some ground-truth of prior image data.

In this work, the goal is to compress the ground-truth image data into a neural model which maps live camera footage to geolocation coordinates.

Twitter user Stephan Sturges demonstrates his solution<d-cite key="Sturges_2023"></d-cite> for allowing a drone with a downward-facing camera to geolocate through cross-referencing against a database of satellite images:

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/2023-11-09-dof-visual-place-recognition-satellite/sturges_satellite_vpr.jpeg" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Twitter user Stephan Sturges shows the results<d-cite key="Sturges_2023"></d-cite> of geolocation based on Visual Place Recognition.
</div>

The author of the above tweet employs a reference database of images. It would be interesting to eliminate the need for a raw dataset.

Thus, this works seeks to develop a neural network which maps a terrain image from the agent's downward-facing camera, to a 6-DOF (position/rotation) representation of the agent in 3-space. Hopefully the neural network is more compact than the dataset itself - although aggressive DNN compression will not be a focus of this work.

# Background

The goal-statement - relating a camera image to a location and orientation in the world - has been deeply studied in computer vision and rendering<d-cite key="Anwar_2022"></d-cite>:

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/2023-11-09-dof-visual-place-recognition-satellite/camera_intrinsic_extrinsic.png" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Camera parameters, as described in<d-cite key="Anwar_2022"></d-cite>.
</div>

Formally<d-cite key="Anwar_2022"></d-cite>,
* The image-formation problem is modeled as a camera forming an image of the world using a planar sensor.
* **World coordinates** refer to 3-space coordinates in the Earth or world reference frame.
* **Image coordinates** refer to 2-space planar coordinates in the camera image plane.
* **Pixel coordinates** refer to 2-space coordinates in the final image output from the image sensor, taking into account any translation or skew of pixel coordinates with respect to the image coordinates.

The mapping from world coordinates to pixel coordinates is framed as two composed transformations, described as sets of parameters<d-cite key="Anwar_2022"></d-cite>:
* **Extrinsic camera parameters** - the transformation from world coordinates to image coordinates (affected by factors "extrinsic" to the camera internals, i.e. position and orientation.)
* **Intrinsic camera parameters** - the transformation from image coordinates to pixel coordinates (affected by factors "intrinsic" to the camera's design.)

And so broadly speaking, this work strives to design a neural network that can map from an image (taken by the agent's downward-facing camera) to camera parameters of the agent's camera. With camera parameters in hand, geolocation parameters automatically drop out from extracting extrinsic translation parameters.

To simplify the task, assume that camera intrinsic characteristics are consistent from image to image, and thus could easily be calibrated out in any application use-case. Therefore, this work focuses on inferring **extrinsic camera parameters** from an image. We assume that pixels map directly into image space.

The structure of extrinsic camera parameters is as follows<d-cite key="Anwar_2022"></d-cite>:

$$
\mathbf{E}_{4 \times 4} = \begin{bmatrix} \mathbf{R}_{3 \times 3} & \mathbf{t}_{3 \times 1} \\ \mathbf{0}_{1 \times 3} & 1 \end{bmatrix}
$$

where $$\mathbf{R}_{3 \times 3} \in \mathbb{R^{3 \times 3}}$$ is rotation matrix representing the rotation from the world reference frame to the camera reference frame, and $$\mathbf{t}_{3 \times 1} \in \mathbb{R^{3 \times 1}}$$ represents a translation vector from the world origin to the image/camera origin.

Then the image coordinates (a.k.a. camera coordinates) $$P_c$$ of a world point $$P_w$$ can be computed as<d-cite key="Anwar_2022"></d-cite>:

$$
\mathbf{P_c} = \mathbf{E}_{4 \times 4} \cdot \mathbf{P_w}
$$

# Proposed solution

## Image-to-extrinsics encoder architecture

The goal of this work, is to train a neural network which maps an image drawn from $$R^{3 \times S \times S}$$ (where $$S$$ is pixel side-length of an image matrix) to a pair of camera extrinsic parameters $$R_{3 \times 3}$$ and $$t_{3 \times 1}$$:

$$
\mathbb{R^{3 \times S \times S}} \rightarrow \mathbb{R^{3 \times 3}} \times \mathbb{R^3}
$$

The proposed solution is a CNN-based encoder which maps the image into a length-12 vector (the flattened extrinsic parameters); a hypothetical architecture sketch is shown below:

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/2023-11-09-dof-visual-place-recognition-satellite/nn.svg" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Image encoder architecture.
</div>

## Data sources for offline training

Online sources<d-cite key="Geller_2022"></d-cite> provide downloadable satellite terrain images.

## Training and evaluation

The scope of the model's evaluation is, that it will be trained to recognize aerial views of some constrained area i.e. Atlantic City New Jersey; this constrained area will be referred to as the "area of interest."

### Data pipeline

The input to the data pipeline is a single aerial image of the area of interest. The output of the pipeline is a data loader which generates augmented images.

The image of the area of interest is $$\mathbb{R^{3 \times T \times T}}$$ where $$T$$ is the image side-length in pixels.

Camera images will be of the form $$\mathbb{R^{3 \times S \times S}}$$ where $$S$$ is the image side-length in pixels, which may differ from $$T$$.

* **Generate an image from the agent camera's vantage-point**
    * Convert the area-of-interest image tensor ($$\mathbb{R^{3 \times T \times T}}$$) to a matrix of homogenous world coordinates ($$\mathbb{R^{pixels \times 4}}$$) and an associated matrix of RGB values for each point ($$\mathbb{R^{pixels \times 3}}$$)
        * For simplicity, assume that all features in the image have an altitutde of zero
        * Thus, all of the pixel world coordinates will lie in a plane
    * Generate random extrinsic camera parameters $$R_{3 \times 3}$$ and $$t_{3 \times 1}$$
    * Transform the world coordinates into image coordinates ($$\mathbb{R^{pixels \times 3}}$$) (note, this does not affect the RGB matrix)
    * Note - this implicitly accomplishes the commonly-used image augmentations such as shrink/expand, crop, rotate, skew
* **Additional data augmentation** - to prevent overfitting
    * Added noise
    * Color/brightness adjustment
    * TBD
* **Convert the image coordinates and the RGB matrix into a camera image tensor ($$\mathbb{R^{3 \times S \times S}}$$)**

Each element of a batch from this dataloader, will be a tuple of (extrinsic parameters,camera image).

## Training

* For each epoch, and each mini-batch...
* unpack batch elements into camera images and ground-truth extrinsic parameters
* Apply the encoder to the camera images
* Loss: MSE between encoder estimates of extrinsic parameters, and the ground-truth values

### Hyperparameters
* Architecture
    * Encoder architecture - CNN vs MLP vs ViT(?) vs ..., number of layers, ...
    * Output normalizations
    * Nonlinearities - ReLU, tanh, ...
* Learning-rate
* Optimizer - ADAM, etc.
* Regularizations - dropout, L1, L2, ...

## Evaluation

For a single epoch, measure the total MSE loss of the model's extrinsic parameter estimates relative to the ground-truth. 

## Feasibility

Note that I am concurrently taking 6.s980 "Machine learning for inverse graphics" so I already have background in working with camera parameters, which should help me to complete this project on time.