---
layout: distill
title: Graph Diffusion for Articulated Objects - Project Proposal
description: In the fields of robotics and computer graphics, learning how to generate articulated objects that look and function accurately to the real world. The conditional generation of CAD/URDF models will be a significant advantage in the field of Real2Sim and is a crucial step to enabling generalizable robotics in the real world. Recent advancements in generative models, including diffusion, have opened up the possibilities of work in the supervised generation of data, ranging from images to molecular and even robot action information. This project explores the feasibility of the conditional generation of URDF data conditioned on a text prompt, leveraging graph neural networks to encode spatial/kinematic constraints.
date: 2023-11-16
htmlwidgets: true

# Anonymize when submitting
# authors:
#   - name: Anonymous

authors:
  - name: Anirudh Valiveru
    url: "https://anirudhv27.github.io/"
    affiliations:
      name: CSAIL, MIT

bibliography: 2023-11-16-project-proposal.bib  

# Add a table of contents to your post.
#   - make sure that TOC names match the actual section names
#     for hyperlinks within the post to work correctly.
toc:
  - name: Project Vision
  - name: Outline of Steps

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

## Project Vision
Recent advancements in generative AI have transformed robotic capabilities across all parts of the stack, whether in control, planning, or perception. As self-driving cars roll out to public roads and factory assembly-line robots become more and more generalizable, embodied intelligence is transforming the way that humans interact with each other and automate their daily tasks. 

Across the robotic manipulation stack, I am most interested in exploring the problem of perception; using the limited sensors available to it, how can a robot gain a rich understanding of its environment so that it can perform a wide array of general tasks with ease? Developments in inverse graphics, such as NeRF and recent models like PointE or DreamGaussian have allowed roboticists to harness the power of deep learning to make more detailed scene representations, enabling their robots to leverage 3D inputs to perform complicated tasks.

One direction that I have been very interested in exploring is in developing robust representations that accurately represent a scene’s kinematic constraints as well, which will allow robots to make plans and predict the outcomes of their actions in an easier way.

In this vein, I hope to explore the feasibility of using modern generative modeling techniques such as diffusion to generate models that are useful to the user. Since articulated objects can be expressed as graphs, I want to specifically learn graph properties of an object either from a single image or a series of a few frames of a short video, with the goal of generating a URDF of the object at the very end. 

The most recent work in the space is NAP: Neural Articulation Prior<d-cite key="lei2023nap"></d-cite>, which is the first work to explore the use of graph denoising networks to generate URDF. Their work, while an important step in the direction of URDF generation, often generates physically implausible outputs that don't actually represent the ground truth in the best way. This project would be successful if I am able to explore the use of novel loss function/algorithmic innovation to perform better than NAP at real-world scenarios, perhaps one that can also be conditioned on text-based prompting.

## Outline of Steps

1. Collect a dataset of labeled URDF assets with known natural language prompts along with URDF and geometric information. 
2. Reproduce NAP's work and fully understand how it is working, trying it on various cases to get a sense of where the paper's method breaks.
3. The results of this will determine my next steps. Some directions to explore and gain a deeper understanding of the problem include trying new architectures (eg. flow networks), loss functions that incorporate physical constraints more explicitly, or a more data-driven exploration of the efficacy of larger-scale training.
