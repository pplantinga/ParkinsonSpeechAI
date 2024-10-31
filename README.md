# ParkinsonSpeechAI

This repository serves as a collection of recipes and analysis documents serving to organize efforts around detection, classification, and analysis of the speech patterns of Parkinson's patients.

To begin with, we have started analyzing a dataset of speech samples of Parkinson's patients and controls collected by the Quebec Parkison's Network (QPN) including around 200 participants. The participants were asked to complete 5 tasks, listed below:

* **Maximum sustained phonation**: Participants were asked to take a deep breath and to sustain the vowel sound /a/ for as long as possible at a comfortable pitch and loudness in one exhalation. This was repeated three times.
* **Spontaneous speech**: Participants were presented with a picture taken from the Boston Diagnostic Aphasia Exam (Goodglass & Kaplan, 1983) and asked to describe everything that is going on in the picture.
* **Paragraph reading**: Participants were asked to read a paragraph out loud. Bilingual patients did this in both English and French.
* **Sentence repetition**: Participants were asked to repeat four short sentences. Bilingual patients did this in both English and French.
* **Autobiographical memory**: Participants were asked to describe two specific memories in detail: one from early childhood (up to age 11) and one from the last year.

So far our analysis and experimentation consists of two efforts:

* An analysis effort on the level of vocal features, computed using an in-progress addition to SpeechBrain. The analysis is in the included notebook: [voice-analysis.ipynb](voice-analysis.ipynb), and the SpeechBrain contribution can be found at [PR #2689](https://github.com/speechbrain/speechbrain/pull/2689). 
* A recipe for detection of Parkinsons' using `wav2vec2` or `WavLM` to extract features, and `ECAPA-TDNN` as a classification model. The training script can be seen at [train.py][recipes/QPN/detection/train.py] and the hyperparameters at [wavlm_ecapa.yaml](recipes/QPN/detection/hparams/wavlm_ecapa.yaml)

Eventually, we hope to integrate these two efforts by training `ECAPA-TDNN` or `SincNet` or other classification models on top of traditional voice features to compare their predictiveness on different tasks with the features from pre-trained self-supervised models. In addition, we will explore post-hoc interpretation of these models using `L-MAC` or other post-hoc interpretation methods.
