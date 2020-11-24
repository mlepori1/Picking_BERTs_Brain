import torch
from pytorch_pretrained_bert import BertTokenizer, BertModel
import logging
import matplotlib.pyplot as plt
import sys
import numpy as np
sys.path.append("../..")
import glove_utils.utils as utils
import RSA_utils.utils as RSA
from sklearn.metrics.pairwise import cosine_similarity
from statsmodels.stats.descriptivestats import sign_test
from scipy.stats import ttest_ind, spearmanr
import random
import matplotlib.pyplot as plt

noun_idxs = [1, 5]
pro_idx = 7

noun_list = [
            'doctor',
            'artist',
            'robot',
            'person',
            'dancer',
            'painter',
            'cop',
            'politician',
            'student',
            'teacher',
            'farmer',
            'banker',
            'lawyer',
            'peasant',
            'chef',
            'pilot',
            'athlete',
            'fairy',
            'monster',
            'alien',
            'ghost',
            'vampire',
            'mummy'
            ]

if __name__ == "__main__":

    np.random.seed(seed=9)
    random.seed(9)

    # Preprocess Corpus
    glove_list, bert_list = RSA.preprocess_data('./pronominal_corpus.txt', noun_list)
    print("data processed")

    # Generate dictionary of Glove embedding hypothesis models
    embed_dict = RSA.get_glove_embeds(glove_list, "../../glove_utils/glove/glove.6B.300d.txt", 300, noun_idxs, noun_list, pro_idx)
    glove_ant = np.array(embed_dict[noun_idxs[0]])
    glove_nonant = np.array(embed_dict[noun_idxs[1]])
    glove_rand = np.array(embed_dict[-1])
    print("glove embeds generated")

    # Generate BERT embedding reference model
    bert_embeds = RSA.get_bert_embeds(bert_list, pro_idx + 1)
    print("BERT embeds generated")

    rsa_ant_dist = []
    rsa_nonant_dist = []
    rsa_rand_dist = []

    samples = []

    # Generate 100 samples of representational similarity
    while len(samples) < 100:

        sample = np.random.choice(range(0, len(glove_list)), replace = False, size=200)
        if set(sample) in samples:
            continue
        samples.append(set(sample))

        samp_bert_embeds = bert_embeds[sample]
        samp_glove_ant = glove_ant[sample]
        samp_glove_nonant = glove_nonant[sample]
        samp_glove_rand = glove_rand[sample]

        bert_geom = RSA.calculate_geometry(samp_bert_embeds)
        ant_geom = RSA.calculate_geometry(samp_glove_ant)
        nonant_geom = RSA.calculate_geometry(samp_glove_nonant)
        rand_geom = RSA.calculate_geometry(samp_glove_rand)

        rsa_ant_dist.append(spearmanr([ant_geom, bert_geom], axis=1)[0])
        rsa_nonant_dist.append(spearmanr([nonant_geom, bert_geom], axis=1)[0])
        rsa_rand_dist.append(spearmanr([rand_geom, bert_geom], axis=1)[0])

    # Run tests and generate plots
    print(f'RSA Verb + Antecedent: {np.mean(rsa_ant_dist)} STD: {np.std(rsa_ant_dist)}')
    print(f'RSA Verb + Non-Antecedent: {np.mean(rsa_nonant_dist)} STD: {np.std(rsa_nonant_dist)}')
    print(f'RSA Verb + Random Noun: {np.mean(rsa_rand_dist)} STD: {np.std(rsa_rand_dist)}')


    print(f'Sign Test Non-Antecedent vs. Antecedent: {sign_test(np.array(rsa_ant_dist) - np.array(rsa_nonant_dist))[1]}')
    plt.hist(np.array(rsa_ant_dist) - np.array(rsa_nonant_dist), bins=15)
    plt.title('Differences: Antecedent - Non-Antecedent', fontsize=17)
    plt.xlabel('Difference in Representational Similarity', fontsize=17)
    plt.xlim(-.05, .05)
    plt.ylabel('Count', fontsize=17)
    plt.axvline(x=0.0, color='k', linestyle='--')
    plt.show()
    print(f'Sign Test Random Noun vs. Antecedent: {sign_test(np.array(rsa_rand_dist) - np.array(rsa_ant_dist))[1]}')
    print(f'Sign Test Non-Antecedent Noun vs. Random Noun: {sign_test(np.array(rsa_nonant_dist) - np.array(rsa_rand_dist))[1]}')




