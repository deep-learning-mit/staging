---
layout: page
title: call for blogposts
permalink: /call
description:
nav: true
nav_order: 2
---

> **Announcement**: the submission deadline has been slightly modified. 
> - **February 2nd AOE** is now an *abstract deadline*; please submit this on [OpenReview](https://openreview.net/group?id=ICLR.cc/2023/BlogPosts&referrer=%5BHomepage%5D(%2F)).
> - **February 10th AOE** is the deadline for any modifications to your blog posts (via a [pull request on github](https://github.com/iclr-blogposts/staging/pulls)).
> - **April 28th AOE** is the deadline for the camera-ready submission. Please follow the instructions [here]({{ '/submitting#camera-ready-instructions' | relative_url }}).
>
> See the [submission instructions]({{ '/submitting#submitting-your-blog-post' | relative_url }}) for more details.

# Call for blogposts

We invite all researchers and practicioners to submit a blogpost discussing work previously published at ICLR, to the ICLR 2023 blogpost track.

The format and process for this blog post track is as follows:

- Write a post on a subject that has been published at ICLR relatively recently.
    The authors of the blog posts will have to declare their conflicts of interest (positive nor negative) with the paper (and their authors) they write about. 
    Conflicts of interest include:
    - Recent collaborators (less than 3 years)
    - Current institution.

    Blog Posts must not be used to highlight or advertise past publications of the authors or of their lab. 
    Previously, we did not accept submissions with a conflict of interest, however this year we will only ask the authors to report if they have such a conflict. 
    If so, reviewers will be asked to judge if the submission is sufficiently critical and objective of the papers addressed in the blog post. 

- The posts will be created and published under a unified template; see [the submission instructions]({{ '/submitting' | relative_url }})
    and the [sample post]({{ '/blog/2022/distill-example' | relative_url }}) hosted on the blog of this website.

- Blogs will be peer-reviewed (double-blind) for quality and novelty of the content: clarity and pedagogy of the exposition, new theoretical or practical insights, reproduction/extension of experiments, etc.
We are slightly relaxing the double-blind constraints by assuming good faith from both submitters and reviewers (see [the submission instructions]({{ '/submitting' | relative_url }}) for more details).

## Key Dates

- **Abstract  deadline**: February 2nd AOE, 2023 (submit to [OpenReview](https://openreview.net/group?id=ICLR.cc/2023/BlogPosts&referrer=%5BHomepage%5D(%2F))).
&nbsp;

- **Submission  deadline**: February 10th AOE, 2023 (any modifications to your blog post, via a [pull request on github](https://github.com/iclr-blogposts/staging/pulls)).
&nbsp;

- **Notification of acceptance**: March 31st, 2023
&nbsp;

- **Camera-ready merge**: April 28th, 2023

## Submission Guidelines

> See [the submission instructions]({{ '/submitting' | relative_url }}) for more details.

For this edition of the Blogposts Track, we will forgo the requirement for total anonymity. 
The blog posts **must be anonymized for the review process**, but users will submit their anonymized blog posts via a pull request to a staging repository (in addition to a submission on OpenReview).
The post will be merged into the staging repository, where it will be deployed to a separate Github Pages website. 
Reviewers will be able to access the posts directly through a public url on this staging website, and will submit their reviews on OpenReview.
Reviewers should refrain from looking at the git history for the post, which may reveal information about the authors.

This still largely follows the Double-Blind reviewing principle; it is no less double-blind than when reviewers are asked to score papers that have previously been released to [arXiv](https://arxiv.org/), an overwhelmingly common practice in the ML community.
This approach was chosen to lower the burden on both the organizers and the authors; last year, many submissions had to be reworked once deployed due to a variety of reasons.
By allowing the authors to render their websites to Github Pages prior to the review process, we hope to avoid this issue entirely. 
We also avoid the issue of having to host the submissions on a separate server during the reviewing process.

However, we understand the desire for total anonymity. 
Authors that wish to have a fully double-blind process might consider creating new GitHub accounts without identifying information which will only be used for this track.
For an example of a submission in the past which used an anonymous account in this manner, you can check out the [World Models blog post (Ha and Schmidhuber, 2018)](https://worldmodels.github.io/) and the [accompanying repository](https://github.com/worldmodels/worldmodels.github.io).