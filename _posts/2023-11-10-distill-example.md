---
layout: distill
title: GINTransformer vs. Bias
description: Your blog post's abstract.
  This is an example of a distill-style blog post and the main elements it supports.
date: 2023-11-10
htmlwidgets: true

authors:
  - name: Yeabsira Moges
    url: "https://www.linkedin.com/in/yeabsira-moges/"
    affiliations:
      name: AI-DS, MIT

# must be the exact same name as your blogpost
bibliography: 2023-11-10-distill-example.bib  

# Add a table of contents to your post.
#   - make sure that TOC names match the actual section names
#     for hyperlinks within the post to work correctly.
toc:
  - name: Proposal

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

The first piece of information that a person recieves about a given topic determines their belief as a whole on said topic. This is shown in expirements where participants beliefs on several topics were challenged with empirical evidence against their beliefs. Studies consistently show that one a person has their mind made up, it is significantly more difficult to change their mind everytime you challenge them on it. Every interaction solidifies their belief. This is epseically important in the context of the social media era we are living in. A lot of the time, people's first impressions over a given event gets primed by what they see about it on theif feeds. This is coming to determine more and more discourse, and especially so when global events occur and those under duress can now more broadly share their stories and struggles. While good, we also have to contend with oppositional, orpessive forces using thise to boon their politic. Being able to determine the source of a given topic, or being able to filter through accounts with troublesome history, would bridge the misinformation gap that has always been a problem long before the social networks of the day.

To measure this information flow, I propose using a GIN-Based Transformer implimentation to tackle misinformation detection and tracking. The dataset will be constructed from a few years of social media activity in clusters between active users. While the age dunamics across social media apps vary greatly, I predict that a similar trend in misinformation will appear once we abstract away all the noise. I am choosing to implement this using a GIN because I want to take advantage of the network architectures isomorphism property to create non-sparse dense connections for the transformer network to take advantage of to the fullest with multi-headed attention. Each node in the network will comprise tweets and character profiles attached to them, giving context for the post content. I want to exploit this structure to determine the underlying trends that determine communication online.

Detecting misinformation is hard. The problem on in the internet age is that detecting misinformation is akin to detecting whether a given claim is true or not, esentially lie detection. This, understandably is really difficult to do even with fact checkers because sometimes, there simply is no one that knows what the whole truth is. Instead of trying to tackle misinformation directly, this proposed approach works to analyze underlying trends in the profiles of people that typically engage in spreading misinformation, and the typical structure that said misinformation takes--a metric i define as information density. Information density will serve to measure the level to which there is a correspondence between the models measure of the veracity of a given claim and the models measure of the profile said text came from.

I am hoping to find a robust way to compute the information density of a given account, text pair and use that to determine how trustworthy a given claim is based on previous percieved patterns. In additon to the architecture above, I will be using conditional prompting to augment my data and will finetune my transformer network for the tweets using Distilbert. I want the model to be as light weight and portable as possible, as such I want the predictive ability of my network to not be costly.