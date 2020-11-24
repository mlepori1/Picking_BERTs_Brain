# Picking_BERTs_Brain

Code and Data for Picking BERT's Brain: Probing for Linguistic Dependencies in Contextualized Embeddings Using Representational Similarity Analysis by Michael Lepori and R. Thomas McCoy, presented at COLING 2020

## How to use this repository
* Download the GloVe embeddings that were pretrained on Wikipedia 2014 + Gigaword 5 from this link: https://nlp.stanford.edu/projects/glove/
* Ensure that this folder is named "glove", and place it in the glove_utils directory
* Each directory contains a Test file. Simply run python3 Test_TESTNAME to reproduce the results from the paper

## Notes
* The corpora used in the paper are contained in each test directory, as are the underlying PCFGs used to generate the corpora
* The PCFG corpus generation script is found in the Corpus Generation directory
* The Diagnostic_Classifiers directory contains the code to reproduce the experiments in Appendix A
* The Normality_of_BERT directory contains the code to reproduce the analyses in Appendix B
* Note: Throughout this repo, we use the term "anaphor" instead of "reflexive"

## Contact

If you have any questions or comments about the code, paper, or analysis method please contact Michael Lepori at mlepori19@gmail.com
