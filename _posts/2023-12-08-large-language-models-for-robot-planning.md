---
layout: distill

description: Pre-trained large vision-language models (VLMs), such as GPT4-Vision, uniquely encode relationships and contextual information learned about the world through copious amounts of real-world text and image information. Within the context of robotics, the recent explosion of advancements in deep learning have enabled innovation on all fronts when solving the problem of generalized embodied intelligence. Teaching a robot to perform any real-world task requires it to perceive its environment accurately, plan the steps to execute the task at hand, and accurately control the robot to perform the given task. This project explores the use of vision-language models to generate domain descriptions. These can be used for task planning, closing the gap between raw images and semantic understanding of interactions possible within an environment.

date: 2023-12-08
htmlwidgets: true

# Anonymize when submitting
# authors:
#   - name: Anonymous

authors:
  - name: Anirudh Valiveru
    url: "https://anirudhv27.github.io/"
    affiliations:
      name: CSAIL, MIT

bibliography: 2023-12-08-llms-for-robot-planning.bib

# Add a table of contents to your post.
#   - make sure that TOC names match the actual section names
#     for hyperlinks within the post to work correctly.
toc:
  - name: Project Background
  - name: Related Work
  - name: Experiments and Findings
  - name: Future Work and Implications

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

## Project Background

Recent advancements in generative AI have transformed robotic capabilities across all parts of the stack, whether in control, planning, or perception. As self-driving cars roll out to public roads and factory assembly-line robots become more and more generalizable, embodied intelligence is transforming the way that humans interact with each other and automate their daily tasks.

Across the robotic manipulation stack, I am most interested in exploring the problem of scene representation; using the limited sensors available, how might a robot build a representation of its environment that will allow it to perform a wide range of general tasks with ease? While developments in inverse graphics like NeRF have given robots access to increasingly rich geometric representations, recent work in language modeling has allowed robots to leverage more semantic scene understanding to plan for tasks.

### Introduction to Task Planning

In robotics, the term **task planning** is used to describe the process of using scene understanding to break a *goal* down into a sequence of individual *actions*. This is in contrast with *motion planning*, which describes the problem of breaking a desired *movement* into individual configurations that satisfy some constraints (such as collision constraints). While simply using motion planning to specify a task is necessary for any generalized robotic system, *task planning* provides robots with a *high-level* abstraction that enables them to accomplish multi-step tasks. 

Take the problem of brushing ones teeth in the morning. As humans, we might describe the steps necessary as follows:

1. Walk to the sink.
2. Grab the toothbrush and toothpaste tube.
3. Open the toothpaste tube.
4. Squeeze toothpaste onto brush.
5. Brush teeth.
6. Rinse mouth.
7. Clean toothbrush.
8. Put everything back.

### Planning Domain Definition Language (PDDL) Explained

Creating a task plan is a trivial task for humans. However, a computer must use a state-space search algorithm like *A\* search* to plan a sequence of interactions from a *start state* to a desired *goal state*. Doing so requires us to define a standard that formally specifies all relevant *environment states*, along with the *preconditions* and *effects* of all possible transitions between two states.

The Planning Domain Definition Language (PDDL) was invented to solve this problem. Description languages like PDDL allow us to define the space of all possible environment states using the states of all entities that make up the environment. Environments are defined as a task-agnostic *domain file*, while the *problem file* defines a specific task by specifying a desired *start* and *end* state.

[Add a figure here giving a render, a pddl file, and an example problem]

Despite task planning's utility, however, there is one major drawback; this approach to planning requires the robot to have a *detailed PDDL domain file* that accurately represents its environment. Generating this file from perception requires not only a semantic understanding of all objects in a space, but also of all possible interactions between these objects, as well as all interactions that the robot is afforded within the environment. Clearly, there is a major gap between the task-planning literature and the realities of upstream perception capabilities.

The rapid advancement of LLMs and vision-language models open up a world of possibilities in closing this gap, as robotic perception systems may be able to leverage learned world understanding to generate PDDL files of their own to use in downstream planning tasks. This project aims to investigate the question: can VLMs be used to generated accurate PDDL domains?

## Related Work

[Add one related works figure]

The use of LLMs in robotic planning and reasoning has exploded in the past few years, due to the promise of leveraging a language model's internal world understanding to provide more information for planning. One such work is LLM+P<d-cite key=""></d-cite>, which combines an LLM with a classical planner to solve a given problem specified in natural language, using PDDL as an intermediate representation. LLM+P works by converting the description into a into a PDDL problem representation, running a classical planning algorithm to find a solution, and then computing the sequence of actions back into a natural language description interpretable by humans. Importantly, LLM+P demonstrates that using an LLM to output a PDDL representation can be a viable strategy in solving planning problems that are specified to a robot. However, there are a few limitations. For one, LLM+P assumes that a relevant domain file is already provided to the robot, specifying all entities and their relationships within the environment's context. While domain files are generally carefully crafted by hand, vision-language models can automate this process.

LLMs have also been used to solve plans directly, to varying levels of success. Works like SayCan[] and LLM-Planner[] use the LLM as a planning engine directly, circumventing the need to use a traditional high-level planner completely. SayCan, in particular, uses a combination of language-grounded instructions and task affordances that indicate the robot's ability to execute a given task, using language to determine the most viable skill to execute from a set of pre-defined skills. These bodies of work have greatly enabled the ability of robots to parse, understand, and execute instructions given to them by their operators as natural language. Particularly, an LLM's ability to break a problem down into several constituent steps is critical to enabling long-horizon task planning with multiple steps.

Language is an increasingly promising modality for robots to operate in, due to the ubiquity of relevant language data to learn real-world entity relations from the internet. However, foundation models that integrate vision and robot-action modalities enable even stronger semantic reasoning. Google's Robot Transformer 2 (RT-2), for example, is a recent work that performs perception, planning, and control all in a single neural network, leveraging internet-scale data. One major drawback of visuomotor policies, such as that employed by RT-2, is that we lose interpretability of a robot's internal representation.

Nonetheless, multi-modal foundation models have proven to be a useful tool across the spectrum of robotic planning. My project takes inspiration from the above works in LLMs for planning and extends the idea to domain-generation, allowing task-planners to work in real-world scenarios.

## Experimental Setup

[Experimental Setup Figures 1 and 2]

## Results

[Table]

[Bar Chart]

## Discussion

- Implications of our study
- Future Work

### Limitations of our study

## Conclusion