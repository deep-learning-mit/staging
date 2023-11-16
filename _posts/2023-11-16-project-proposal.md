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
  - name: Related Work
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

In this vein, I hope to explore the feasibility of incorporating graphical information to generate articulated URDF models that can be used in downstream robotics applications. Since articulated objects can be expressed as graphs, I want to specifically learn graph properties of an object either from a single image or a series of a few frames of a short video, with the goal of generating a URDF of the object at the very end.

## Related Work

The first work to explore the use of graph denoising networks to generate URDF is NAP: Neural Articulation Prior<d-cite key="lei2023nap"></d-cite>, which conditions its generation on either the object's structural graph or a representation of its partial geometry. Their work, while an important step in the direction of URDF generation, often generates physically implausible outputs that don't actually represent the ground truth in the best way. Other works, such as URDFormer, use a transformer architecture to train on a large dataset of procedurally generated/annotated pairs of URDFs with corresponding images, training a model that can generate statistically accurate URDF models that roughly align with an image given to the model as input.

NAP and URDFormer both generate realistic models that can be used as simulation assets, but struggle to generate an accurate model of real-world 3D data, which is core to closing the Real2Sim gap. Closest to my goal is Ditto, which learns an implicit neural-representation for a point cloud before and after being moved, constructing the URDF representation using a learned correspondence between frames. Ditto's approach using multiple frames to make its reconstruction is critical, because articulation models are inherently ambiguous without information about joint constraints.

However, their main drawback is their assumption of segmenting a point cloud into only two parts. More complicated objects, such as cupboards with handles or multiple drawers, are not supported by their method, which leaves room to explore methods that can infer the whole kinematic tree. To this end, I hope to explore graph-based approaches that are more easily able to extend a method like Ditto to more complicated objects.

This project would be successful if I am able to explore the use of novel loss function/algorithmic innovation to perform better than NAP or Ditto at real-world scenarios, perhaps one that can also be conditioned on text-based prompting or using priors from VLMs like GPT4-Vision.

## Outline of Steps

1. Collect a dataset of labeled URDF assets with known natural language prompts along with URDF and geometric information. 
2. Reproduce Ditto's work and fully understand how it is working, trying it on various cases to get a sense of where the paper's method breaks.
3. Reproduce NAP's work and figure out how it encodes and learns kinematic structure.
4. Make adjustments to Ditto's framework of URDF generation. This will likely involve slightly modifying Ditto's architecture to support graph-based intermediate representations instead of solely working in the realm of unstructured point clouds. Another approach may be to incorporate GPT4-Vision or other pre-trained image-based priors to segment images into prospective rigid bodies. Depending on the results, this project may provide valuable insights into the pros and cons of either approach when extending Ditto to a general multi-link setting.