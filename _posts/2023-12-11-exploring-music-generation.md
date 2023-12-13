---
layout: distill
title: "Exploring Methods for Generating Music"
description: Explores various machine learning techniques for generating music. Compares the performance of traditional RNNs, LSTMs, and transformers on generating sample sequences of music.
date: 2023-12-11
htmlwidgets: true

# Anonymize when submitting
# authors:
#   - name: Anonymous

authors:
  - name: Anonymous
    url:
    affiliations:
      name: MIT

# must be the exact same name as your blogpost
bibliography: 2023-12-11-exploring-music-generation.bib  

# Add a table of contents to your post.
#   - make sure that TOC names match the actual section names
#     for hyperlinks within the post to work correctly.
toc:
  - name: "Introduction"
  - name: "Related Work"
  - name: "Methodology"
  - name: "Results/Experiments"
  - name: "Closing Thoughts and Future Work"

---

# Introduction
The problem of music generation has been widely explored for a long time. Music has very similar parallels to how speech and language is structured. Just like language, music is temporal and in the traditional western sense, has a defined set of standards/rules for how music should be structured. What makes music generation a more challenging problem than language is that music has an artistic/expressive component as well as both low and high level structure. For "good" music, it isn't enough to simply generate a series of notes and harmonies that obey music theory conventions. At the low level, "good" music makes use of varying dynamics, note groupings, and articulation. At the high level, "good" music may feature overarching motifs and specific [forms](https://en.wikipedia.org/wiki/Musical_form) (round, sonata form, ABAB, etc). 
This level of complexity is analagous to the problem of generating poetry and generating speech that mimics a human reciting it. The poetry will have structures like
rhyme, rhythm, motifs, metaphors, etc. and the speech reading it will have to replicate expressiveness to be convinving. This level of complexity is not yet achievable with high
level of robusts by current speech generators, LLMs, and NLP methods. 

It is this level of structural complexity required for generating "good" music that make machine learning methods, specifically deep learning, a compelling approach to 
tackling the problem of generating "good" music. Deep learning methods should be able to capture music's low level music theory structure as well as the high level
It is the hope that given enough data and the right architectures, music generation will be able to mimick a level akin to the best human composers. While music generation such as OpenAi's jukebox <d-cite key="dhariwal2020jukebox"></d-cite> as yielded very good results, it is trained on pure audio frequencies. I will focus on musical generation and training from a "written" / musical structural perspective rather than audio. (Think human speech vs. language/text), as I think this can provide greater insight into how these models learn and what about musical structure is being learned.

# Related Work

There has been several studies/project done in the generation of music. OpenAi has done work with using audio samples to generate music. They took a representation learning and autoencoder approach leveraging VQ-VAEs. Other work <d-cite key="doi:10.1080/25765299.2019.1649972"></d-cite> took approaches similar to me and tried to analyze the "written" structure of music and used a combination of LSTMs and a midi encoding scheme to . Work has been done to capture the expressitivity of music <d-cite key="10124351"></d-cite>, where they leverage large transformer models and condition them on emotion to generate music. There has been success in generating expressitivity based on this conditional approach. My work here aims to analyze purely the syntactical structure of music and will not be leveraging conditioning. 

# Methodology
Before music can be learned and generated, it needs to first be converted to a format that can be input to a ML model. To achieve this I used a subset of a piano midi dataset <d-cite key="ferreira_aiide_2020"></d-cite> and utilized a [program](https://pypi.org/project/py-midicsv/) to convert from MIDI to .csv. Using this .csv file I encoded each note in the midi to a 107 dimensional vector. Where the first 106 dimensions correspond to midi-notes [A0-G9](https://www.inspiredacoustics.com/en/MIDI_note_numbers_and_center_frequencies), and the last dimension is encodes the duration of the midi-note divided by the midi-clock/quarter frequency to get a duration of the note in quarter notes. Since note A0 corresponds to midi-note 21, all of the midinote values are subtracted by this baseline value when being encoded into the vector. If a midi-note is played it is encoded as "ON" in the .csv and as such is represented with a 1 in it's corresponding index in the note vector. For example, if a C4 and A4 note (MIDI note 60, and 69 respectively) are played at the same time in a song, it will be encoded as a 107 dimensional zero vector with indices 37, 47 (60 (midi value) -21 (baseline)-1 (0-index notation)) being 1 and index 106 being the duration of the chord.

I then tested 3 different models to see how they performed. The first model I tested was an RNN with hidden_size = 64, RNN_layers = 2, and sequences of 24, 48, 64, and 200. I next tested LSTM models with hidden_size = 64, RNN_layers = 2, and sequences of 24, 48, 64, and 200 and compared a birection vs. single directional model. The last model I analyzed was a transformer. In which I first took my note encodings and created an embedded representation of the notes and combined this with positional encoding in the sequence of music to get my final embedding to pass into my transformer architecture. 

# Results/Experiments
I found that the RNN architecture to be the worst performing model. It has a high ringing for some training and mostly unstructured and random. The results of a sample music generation can be found [here](https://drive.google.com/drive/folders/1FiuobbyVUnwpUZUx_PYBR57qOwj5jYXe?usp=sharing). The LSTM model took longer to train but performed better with hidden size = 64, sequence_length=48, and 30 epochs. I found that it worked even better when using a bidirectional architecture. A sample generation can be found [here](https://drive.google.com/drive/folders/10CzuEbuVXKCyLsY5vwQZjSKJT1ABqXbA?usp=sharing) in which it was fed the starting 10 notes of Polonaise in A-flat major, Op. 53 and was asked to generate a long sequence from that. The transformer took the longest to train and its results can be found [here](https://drive.google.com/drive/folders/1fGe7xUZyFNlFGMbGB8aXnVfSEx067ZaA?usp=sharing)


# Closing Thoughts and Future Work

As expected the base RNN architecture failed to generate anything meaningful. It took a while to find hyperparameters that would make the LSTM generate something of note, but when it did successfully generate music I was surprised by some of the resemblences it had to music in the training data. 

One noticeable flaw in my work is that I that my metric for success outside of training error is qualitative. It would have been useful for evaluation of my model implementations if I had a quanititative metric. I originally calculated the loss of my models based on how they replicated unseen music from a test set given sequences from the same music, however losses for every model failed to converge in a reasonable amount of time. It is certainly difficult to tell if poor performance is due to implementation or a small dataset and limited compute resources. 

Continuing on the idea of lack of data. One of the challenges I faced was in the curation of my dataset. I originally was going to generate music tokens for my network based on a very descriptive musical format cally [lilypond](https://lilypond.org/). However, there were inconsisencies between samples of music in how they were resprented in the lilypond text format, so creation of a program to transcribe the text to a good format for representing music was very difficult which is why I turned to the more standardized MIDI file format. It is unfortunate because a lot of the the complex expression in music is lost in midi format, making it harder if not impossible for models trained on midi input to learn these complex representations/behavior. I say impossible because if data for musical expression is completely absent from training, then this important component of music is simply out of distribution and impossible to learn. So a better way to encode/represent music is needed for better results.

Moving forward, it would be interesting to explore how representation learning can be used to enhance the generation of music. I wanted to explore the use of VAEs and some of the more advanced variations like the one in used in OpenAi's jukebox, VQ-VAE. These methods maybe be able to capture both the high level structure and complex low level structure found in music. I also want to explore methods for encoding the dynamics, articulation, and expression found in music, something I was not able to do this time around. Lastly, exploring a better way to encode and learn the duration of notes would lead to better music generation. 