
# Datasets

This folder contains data (or scripts to process data) for the paper: [1] "Probing for targeted syntactic knowledge through grammatical error detection"

## Learner corpora

To recreate the learner corpora:

1. Download the FCE [2] and W&I+LOCNESS [3] data from the [BEA 2019 Shared Task](https://www.cl.cam.ac.uk/research/nl/bea2019st/#data)
2. Configure your source and target paths in paths.py
3. Run process.py to correct non-R:VERB:SVA errors and format the data

---

## Marvin & Linzen stimuli

See datasets/marvin_linzen for the processed evaluation stimuli from [4] Marvin & Linzen (2018)

---

## WikEdits

See datasets/wikedits for the processed train and development sets from [5] Grundkiewicz and Junczys-Dowmunt (2014)

---

[1] Davis et al. 2022. Probing for targeted syntactic knowledge through grammatical error detection. Proceddings of CoNLL.

[2] Yannakoudakis, Helen and Briscoe, Ted and Medlock, Ben, ‘A New Dataset and Method for Automatically Grading ESOL Texts’, Proceedings of the 49th Annual Meeting of the Association for Computational Linguistics: Human Language Technologies.

[3] Christopher Bryant, Mariano Felice, Øistein E. Andersen and Ted Briscoe. 2019. The BEA-2019 Shared Task on Grammatical Error Correction. In Proceedings of the 14th Workshop on Innovative Use of NLP for Building Educational Applications (BEA-2019), pp. 52–75, Florence, Italy, August. Association for Computational Linguistics.

[4] R. Marvin and T. Linzen. 2018. Targeted Syntactic Evaluation of Language Models. Proceedings of EMNLP.

[5] Roman Grundkiewicz and Marcin Junczys-Dowmunt. The WikEd Error Corpus: A Corpus of Corrective Wikipedia Edits and its Application to Grammatical Error Correction. In Advances in Natural Language Processing – Lecture Notes in Computer Science, volume 8686, pages 478–490. Springer.