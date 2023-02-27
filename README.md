# ged-syntax-probing

This repository contains code for the paper: "Probing for targeted syntactic knowledge through grammatical error detection" (Davis et al., CoNLL 2022).

## Setup

conda env create -f environment.yml

## Data

The "datasets" folder contains processed data from [1] [Wiki Edits 2.0](https://github.com/snukky/wikiedits) and [2] [Marvin & Linzen (2018)](https://github.com/BeckyMarvin/LM_syneval).

For the [3] W&I+LOCNESS and [4] FCE, see the README.md in the "datasets" folder.

Each dataset is conll formatted, with one token per line, and where blank lines indicate sentence boundaries. Each line has two columns, the first is the token and the second is the grammatical error label. For this work, there are only two labels: C (for correct) and R:VERB:SVA for replacement-subject-verb-agreement error.

## Steps to replicate the results

1. Run src/encode_datasets.py to encode all three datasets with all the models and save them to disk.
2. Run src/train_all_probes.py to train probes using the learner corpora or the WikEdits data
3. Run src/evaluate_datasets.py to evaluate probes on the Marvin & Linzen evaluation stimuli
4. Run src/analysis/analyse_predictions.py to post-process the Marvin & Linzen results.
5. See src/analysis/plotting to generate the plots from the paper.

You can also run the scripts from steps 1-3 with with two optional arguments, in order to encode/train/eval on a specific model and/or dataset:

- "-m [model name]"
- "-d [dataset name]"

## Results from the paper

The results and plots from the paper are included in the "results" folder. The folders tagged "notobe" are from the second experiment, with the "-mlverbs" tag indicating the results from probes trained without "is" and "are".

## Encoding your own datasets

You can encode your own datasets using src/encoding/encode.py, but you'll need to format it into the conll-format (or modify the data/processor.py to read your own file format.)

## Evaluating your own dataset

Run src/eval_probe.py to evaluate a trained probe on a single file (you don't have to encode the file beforehand).

---

## References

If you use any of the datasets from this work, please cite their original papers:

[1] Roman Grundkiewicz and Marcin Junczys-Dowmunt. The WikEd Error Corpus: A Corpus of Corrective Wikipedia Edits and its Application to Grammatical Error Correction. In Advances in Natural Language Processing – Lecture Notes in Computer Science, volume 8686, pages 478–490. Springer.

[2] R. Marvin and T. Linzen. 2018. Targeted Syntactic Evaluation of Language Models. Proceedings of EMNLP.

[3] Christopher Bryant, Mariano Felice, Øistein E. Andersen and Ted Briscoe. 2019. The BEA-2019 Shared Task on Grammatical Error Correction. In Proceedings of the 14th Workshop on Innovative Use of NLP for Building Educational Applications (BEA-2019), pp. 52–75, Florence, Italy, August. Association for Computational Linguistics.

[4] Yannakoudakis, Helen and Briscoe, Ted and Medlock, Ben, ‘A New Dataset and Method for Automatically Grading ESOL Texts’, Proceedings of the 49th Annual Meeting of the Association for Computational Linguistics: Human Language Technologies.