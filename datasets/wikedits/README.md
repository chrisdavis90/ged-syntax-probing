
## Processed sentences from Grundkiewicz and Junczys-Dowmunt (2014)
---

Pre-processing:

1. Download "wiked-v1.0.en.prepro.tgz" from [Wiki Edits 2.0](https://github.com/snukky/wikiedits)
2. Run [ERRANT v2.3.3](https://github.com/chrisjbryant/errant) on the extracted .cor and .err files
3. Run "edit_restrict_error_type_general.py" to correct every grammatical error except for R:VERB:SVA
4. Run "m2_to_conll.py" to create a .conll version
5. Sample training and development sets

As detailed in the paper, for the first experiment, we sample 1936 (1x) sentences to match the number of sentences from the learner corpora. We sample 5 times to create v1 - v5 versions of the training set.

For the second experiment, we remove the verbs included in the Marvin & Linzen dataset (except for "to be") and increase the training set size to 7744 (4x) and 15488 (8x) sentences. Again, sampled five times.

The sampled training and development sets can be found in this folder.

Bibtex citation:

    @inproceedings{wiked2014,
        author = {Roman Grundkiewicz and Marcin Junczys-Dowmunt},
        title = {The WikEd Error Corpus: A Corpus of Corrective Wikipedia Edits and its Application to Grammatical Error Correction},
        booktitle = {Advances in Natural Language Processing -- Lecture Notes in Computer Science},
        editor = {Adam Przepiórkowski and Maciej Ogrodniczuk},
        publisher = {Springer},
        year = {2014},
        volume = {8686},
        pages = {478--490},
        url = {http://emjotde.github.io/publications/pdf/mjd.poltal2014.draft.pdf}
    }