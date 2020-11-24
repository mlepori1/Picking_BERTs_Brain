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
import os

noun_idxs = [1, 4]
verb_idx = 5

noun_list = [
            'boy',
            'girl',
            'man',
            'woman',
            'guy',
            'doctor',
            'artist',
            'robot',
            'person',
            'painter',
            'cop',
            'student',
            'teacher',
            'lawyer',
            'peasant',
            'chef',
            'pilot',
            'athlete',
            'farmer',
            'boys',
            'girls',
            'men',
            'women',
            'guys',
            'doctors',
            'artists',
            'robots',
            'people',
            'painters',
            'cops',
            'students',
            'teachers',
            'lawyers',
            'peasants',
            'chefs',
            'pilots',
            'athletes',
            'farmers',
            'house',
            'building',
            'chair',
            'table',
            'door',
            'window',
            'plane',
            'car',
            'truck',
            'houses',
            'buildings',
            'chairs',
            'tables',
            'doors',
            'windows',
            'planes',
            'cars',
            'trucks'
            ]


if __name__ == "__main__":

    np.random.seed(seed=9)
    random.seed(9)

    # Preprocess corpus
    glove_list, bert_list = RSA.preprocess_data('./copula_PP_corpus.txt', noun_list)
    print("data processed")

    # Get dictionary of Glove embedding hypothesis models
    embed_dict = RSA.get_glove_embeds(glove_list, "../../glove_utils/glove/glove.6B.300d.txt", 300, noun_idxs, noun_list, verb_idx)
    glove_subj = np.array(embed_dict[noun_idxs[0]])
    glove_nonarg = np.array(embed_dict[noun_idxs[1]])
    glove_rand = np.array(embed_dict[-1])
    print("glove embeds generated")

    # Get BERT embedding reference models
    bert_embeds = RSA.get_bert_embeds(bert_list, verb_idx + 1)
    print("BERT embeds generated")

    rsa_subj_dist = []
    rsa_nonarg_dist = []
    rsa_rand_dist = []

    samples = []

    # Generate 100 samples of representational similarity
    while len(samples) < 100:

        sample = np.random.choice(range(0, len(glove_list)), replace = False, size=200)
        if set(sample) in samples:
            continue
        samples.append(set(sample))

        samp_bert_embeds = bert_embeds[sample]
        samp_glove_subj = glove_subj[sample]
        samp_glove_nonarg = glove_nonarg[sample]
        samp_glove_rand = glove_rand[sample]

        bert_geom = RSA.calculate_geometry(samp_bert_embeds)
        subj_geom = RSA.calculate_geometry(samp_glove_subj)
        nonarg_geom = RSA.calculate_geometry(samp_glove_nonarg)
        rand_geom = RSA.calculate_geometry(samp_glove_rand)

        rsa_subj_dist.append(spearmanr([subj_geom, bert_geom], axis=1)[0])
        rsa_nonarg_dist.append(spearmanr([nonarg_geom, bert_geom], axis=1)[0])
        rsa_rand_dist.append(spearmanr([rand_geom, bert_geom], axis=1)[0])


    # Perform tests and generate plots
    print(f'RSA Verb + Subject: {np.mean(rsa_subj_dist)} STD: {np.std(rsa_subj_dist)}')
    print(f'RSA Verb + Non-Argument: {np.mean(rsa_nonarg_dist)} STD: {np.std(rsa_nonarg_dist)}')
    print(f'RSA Verb + Random Noun: {np.mean(rsa_rand_dist)} STD: {np.std(rsa_rand_dist)}')


    print(f'Sign Test Non-Argument vs. Subject: {sign_test(np.array(rsa_subj_dist) - np.array(rsa_nonarg_dist))[1]}')
    plt.hist(np.array(rsa_subj_dist) - np.array(rsa_nonarg_dist), bins=15)
    plt.title('Differences: Subject - Object of the Preposition', fontsize=17)
    plt.xlabel('Difference in Representational Similarity', fontsize=17)
    plt.xlim(-.1, .1)
    plt.ylabel('Count', fontsize=17)
    plt.axvline(x=0.0, color='k', linestyle='--')
    plt.show()
    print(f'Sign Test Random Noun vs. Subject: {sign_test(np.array(rsa_rand_dist) - np.array(rsa_subj_dist))[1]}')
    print(f'Sign Test Non-Argument Noun vs. Random Noun: {sign_test(np.array(rsa_nonarg_dist) - np.array(rsa_rand_dist))[1]}')




