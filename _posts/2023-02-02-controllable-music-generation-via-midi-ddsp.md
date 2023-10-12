---
layout: distill
title:  Controllable Music Generation via MIDI-DDSP
description: Recently, there has been a lot of fascinating work focusing on making music generation for a larger and more general audience. However, these models could be of great help to the artists if they can intervene into the generation process at multiple levels in order to control what notes are played and how they are performed. We hence dive deeper into MIDI-DDSP that helps with high-fidelity generations using an interpretable hierarchy with several degrees of granularity.
date: 2023-02-02
htmlwidgets: true

# Anonymize when submitting
authors:
  - name: Anonymous
#
# must be the exact same name as your blogpost
bibliography: 2023-02-02-controllable-music-generation-via-midi-ddsp.bib  

# Add a table of contents to your post.
#   - make sure that TOC names match the actual section names
#     for hyperlinks within the post to work correctly.
toc:
  - name: Historical background and Motivation
  - name: Notation used
  - name: MIDI - DDSP   
    subsections:
    - name: What is DDSP?
  - name: MIDI-DDSP architecture(summary)
  - name: Experiments
  - name: Discussion and Conclusion
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

## Historical background and Motivation

Early years of synthesizer: Robert Moog, an American engineer developed a synthesizer in which created and shaped sound connected by patch cords where pitch was controlled via voltage. It was popularized in the late 1960s by rock and pop acts such as the Doors, the Grateful Dead, the Rolling Stones, and the Beatles.<d-cite key="wikipedia_2023"></d-cite> During the same time, American engineer Don Bul created the Buchla Modular Electronic Music System Buchla Modular Electronic Music System in which instead of a traditional keyboard he used touchplates where depending on finger position and force voltage was transmitted. However, Moog’s Synthesizer became more accessible and marketable to musicians during 1964 and the mid-1970s.
The earliest versions of synthesizers could only produce a single note at a time. Tom Oberheim, an American engineer, developed some of the early commercial polyphonic synthesizers. The first fully programmable polyphonic synthesizer, - Prophet5 was released in 1978 which used microprocessors to store sounds in patch memory. This allowed synthesisers to go from producing uncertain sounds to "a conventional set of familiar sounds." After introduction of MIDI in 1982, synthesizer market grew dramatically<d-cite key="vail2014synthesizer"></d-cite>.

#### Notation used

DDSP - Differentiable Digital Signal Processing <br />


## MIDI - DDSP
While generative models are function approximator and may assist the development of samples across many domains, this expressiveness comes at the expense of interaction, since users are often limited to black-box input-output mappings without access to the network's internals.<d-cite key="wu2021midi"></d-cite>. This makes sense as we know that having access to the latent space can be very useful for the generative models. Diffusion models lack this!<br />
In computer vision and speech there has been development in methods where users are allowed to interact througout the hierarchy of system making it optimize for realism and control. Whereas in music synthesis methods still lack this interaction in hierarchy of music generation. Recent research states that one can either generate full-band audio or have control of pitch, dynamics and timbre but not both.<d-cite key="hawthorne2018enabling"></d-cite> <d-cite key="wu2021midi"></d-cite>. Authors of MIDI-DDSP<d-cite key="wu2021midi"></d-cite> take inspiration from process of creating music and propose a generative model of music generation organised in a hierarchy for more realism and control. As traditional synthesizer use MIDI standard audio files, MIDI - DDSP translates note timing, pitch, and expression data into granular control of DDSP synthesiser modules.

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/2023-02-02-controllable-music-generation-via-midi-ddsp/MIDI_intro.png" class="img-fluid"  %}
        <em>Figure 1: Hierarchy in music synthesis <d-cite key="engel2020ddsp"></d-cite></em>
    </div>
</div>
It has 3 level hierarchy:notes, performance, synthesis as shown in above figure <br/>
Notes : Similar to how composer writes series of Notes <br/>
Performance : Similar to how performer articulates these notes into dynamics, and expression in music. <br/>
Synthesis : Similar to how the expression are then converted to audio by short-time pitch and timbre changes of the physical vibration. <br/>

MIDI-DDSP can be viewed similarly to a multi-level autoencoder. I has 3 separately trainable modules (DDSP Inference, Synthesis Generator, Expression Generator)<br />
**DDSP Inference** - The DDSP Inference module learns to make predictions about synthesis parameters from audio and then applies those learnings to resynthesized audio using an audio reconstruction loss.<br />
**Synthesis Generator** - The Synthesis Generator module makes predictions regarding synthesis parameters based on notes and the expression qualities associated with those notes. These predictions are then iterated through the use of a reconstruction loss and an adversarial loss. <br />
**Expression Generator** - The Expression Generator module uses autoregressive modelling to provide predictions about note expressions based on a given note sequence which is trained via teacher forcing.
<br />

### BUT what is DDSP!
**Challenges of neural audio synthesis and how DDSP overcomes it** <br />
As shown in below Figure 1(left) shows that strided convolution models generate waveform with overlapping frames and suffer from phase alignment problem. Here phase alignment comes from recording of the same source made with 2 or mics placed at different distance. This distance variation cause the sound to arrive at mics at slightly different times. Figure 1(center) shows spectral leakage which occurs when the Fourier basis frequencies do not completely match the audio, where sinusoids at several nearby frequencies and phases need to be blended to represent a single sinusoid. Although the three waveforms on the right side of Figure 1 appear to have the same sound (a relative phase offset of the harmonics), an autoregressive model would find them to have very different losses. This represent inefficiency of model such that waveform shape does not correspond to perception.
<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/2023-02-02-controllable-music-generation-via-midi-ddsp/challenges.png" class="img-fluid"  %}
        <em>Figure 3: Challenges in neural audio synthesis <d-cite key="engel2020ddsp"></d-cite></em>
    </div>
</div>
DDSP model overcomes above challenge and gain an advantage from the inductive bias of using oscillators while preserving the expressive power of neural networks and end-to-end training.
<d-cite key="engel2020ddsp"></d-cite>
**Why making the synthesis differentiable is important?**
The harmonics-plus-noise model, a differentiable additive synthesis model, generates audio in the paper. A sinusoids-plus-noise model version. The harmonics-plus-noise model is a synthesiser, but it requires specifying each harmonic's amplitude and the filter's frequency response. The harmonics plus-noise synthesiser accurately recreates actual instrument sounds, but its complex synthesis settings prevent direct engagement.<d-cite key="masuda2021synthesizer"></d-cite>
<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/2023-02-02-controllable-music-generation-via-midi-ddsp/ddsp.gif" class="img-fluid"  %}
        <em> Figure 4:Decomposition of a clip of solo violin.</em>
    </div>
</div>
As seen in above animation, the signals for loudness and fundamental frequency are taken from the original audio. As a result of the impacts of the room acoustics, the loudness curve does not reveal clearly differentiated note segmentations. These conditioning signals are input into the DDSP autoencoder, which then makes predictions about amplitudes, harmonic distributions, and noise magnitudes. The entire resynthesis audio is produced by applying the extracted impulse response to the synthesiser audio.

3 main design components:
**Expressive** - due to more params in the synthesizer (also the ability to control the generation process)
**Interpretable** - because of the harmonic oscillator assumption i.e. relying upon fundamental frequency and loudness
**Adaptable** - interpolation between different instruments

## MIDI-DDSP architecture(summary)

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/2023-02-02-controllable-music-generation-via-midi-ddsp/hierachy_generation.png" class="img-fluid"  %}
        <em>Figure 2: MIDI-DDSP modules <d-cite key="engel2020ddsp"></d-cite></em>
    </div>
</div>

This figure gives a high level view of the MIDI-DDSP modules. We also show that full-stack automatic music generation is possible when MIDI-DDSP is combined with a pretrained note generating model.

### Expression Generator

Expression Generator is mainly an autoregressive RNN that is trained to predict "expression controls" from the note sequence. These are the synthesis parameters that will be used in the next network; the synthesis generator.

So what are these Expression Controls that we speak of, you might ask!
These controls also represents few of the choices that the performer makes while performing a composed track. Following is the list of controls applied:

*Volume*: Controls how loud a note is.
*Volume fluctuation*: Controls how the loudness of a note changes over the note.
*Volume peak position*: Controls the location of the peak volume during the course of a note.
*Vibrato*: Controls the degree of a note's vibrato, where Vibrato is a musical effect or a technique where a note changes pitch subtly and quickly
*Brightness*: increases in value produce louder high-frequency harmonics, which in turn controls the timbre of the note.
*Attack Noise*: Controls the amount of noise at the note's beginning.


### Synthesis Generator
Synthesis Generator again is an autoregressive RNN used to predict fundamental frequency, given a conditioning sequence from the previous module. It might now sound obvious that these params are in turn used in the next module, which is the DDSP Inference.

### Interaction with DDSP interface

⇒ How is this different from DDSP?
CNN is utilised on a logarithmic scale. Mel-spectrograms aid models in obtaining more data from audio input, enabling more precise estimation of synthesis parameter.
A fully linked network is used on fundamental frequency and loudness in our DDSP inference module. In order to extract contextual information, the output is concatenated with the CNN output and sent to the bi-directional LSTM. To map the characteristics to the synthesis settings, another fully connected layer is utilised.

## Experiments
<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/2023-02-02-controllable-music-generation-via-midi-ddsp/Results1.png" class="img-fluid"  %}
        <em>Comparison of reconstruction audio accuracy<d-cite key="wu2021midi"></d-cite></em>
    </div>
</div>
As Shown in above figure in left there is comparison of Mel spectrogram of synthesis results and on right it shows comparison of synthesis quality from listening test. From figure(right) it is seen that MIDI-DDSP inference is perceived as likely as ground truth compared to other methods such as MIDI2Params, Ableton and FluidSynth.  
<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/2023-02-02-controllable-music-generation-via-midi-ddsp/Results.png" class="img-fluid"  %}
    </div>
</div>
Above figure shows pearson correlation between the input control and the respective output quantity. It is seen that there is strong correlation between input control and note expression output.
## Future direction
Author mentions in paper that one promising direction for future research is to apply this method to polyphonic recordings by means of multi-instrument transcription and multi-pitch tracking.
When making differentiable it goes through many combinations and the model explore all the different possibilities and backpropagate through the soft weighting of all those possible paths, which can very soon become intractable and Reinforcement Learning could possibly be used for this search(as it is a combinatorial problem->additive synthesis).
