# ReaseachAbstractTaggingHackathon

Model Used -> SciBert

SciBERT is a BERT model trained on scientific text.

* SciBERT is trained on papers from the corpus of semanticscholar.org. Corpus size is 1.14M papers, 3.1B tokens
* It is trained on the full text of the papers in training,and not just abstracts
* It has its own vocabulary (scivocab) that's built to best match the training corpus

The following is an implementation of SciBert via TensorFlow2

Code File used for submission -> https://github.com/gcspkmdr/ReaseachAbstractTaggingHackathon/blob/master/scibert-wrapped-in-tf2.ipynb

The model has been trained on 4 modifications of data:
* Original
* Original without Latex Tags
* Augmented Data(Translate into some language like dutch + De-Translate back to english)
* Augmented Data - without Latex Tags
The details can be found out in this notebook https://github.com/gcspkmdr/ReaseachAbstractTaggingHackathon/blob/master/text-data-augmentation-latex-tag-translate.ipynb

Accelerator used -> GPU

* An Ensemble is created using 10 fold CV * 4 modification of datasets
* A manual ensembling is done with 40 such probablity files(not in this repo) generated based on which particular ensemble is generating the best LB F1 score
