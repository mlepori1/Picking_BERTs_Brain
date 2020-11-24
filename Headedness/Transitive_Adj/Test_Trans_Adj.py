import torch
from pytorch_pretrained_bert import BertTokenizer, BertModel
import logging
import matplotlib.pyplot as plt
import sys
import numpy as np
sys.path.append("../..")
import glove_utils.utils as utils
import RSA_utils.utils as RSA
from statsmodels.stats.descriptivestats import sign_test
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import ttest_ind, spearmanr
import random

lexical_idxs = [1, 2, 3, 5, 6]

verb_list = ['loves',
            'hates',
            'likes',
            'smells',
            'touches',
            'pushes',
            'moves',
            'sees',
            'lifts',
            'hits']


if __name__ == "__main__":

    np.random.seed(seed=9)
    random.seed(9)

    # Preprocess Corpus
    glove_list, bert_list = RSA.preprocess_data('./head_adj_trans_corpus.txt')
    print("data processed")

    # Generate glove hypothesis models
    embed_dict = RSA.get_glove_embeds(glove_list, "../../glove_utils/glove/glove.6B.300d.txt", 300, lexical_idxs, verb_list)
    adj1 = np.array(embed_dict[lexical_idxs[0]])
    subj = np.array(embed_dict[lexical_idxs[1]])
    verb = np.array(embed_dict[lexical_idxs[2]])
    adj2 = np.array(embed_dict[lexical_idxs[3]])
    obj = np.array(embed_dict[lexical_idxs[4]])
    rand_verb = np.array(embed_dict[-1])
    print("glove embeds generated")

    # Generate BERT reference model
    bert_embeds = RSA.get_bert_embeds(bert_list, 0)
    print("BERT embeds generated")

    rsa_subj_dist = []
    rsa_obj_dist = []
    rsa_verb_dist = []
    rsa_adj1_dist = []
    rsa_adj2_dist = []
    rsa_rand_verb_dist = []


    samples = []

    # Generate 100 samples of representational similarity
    while len(samples) < 100:

        sample = np.random.choice(range(0, len(glove_list)), replace = False, size=200)
        if set(sample) in samples:
            continue

        samples.append(set(sample))

        samp_bert_embeds = bert_embeds[sample]
        samp_subj = subj[sample]
        samp_obj = obj[sample]
        samp_verb = verb[sample]
        samp_adj1 = adj1[sample]
        samp_adj2 = adj2[sample]
        samp_rand_verb = rand_verb[sample]

        bert_geom = RSA.calculate_geometry(samp_bert_embeds)
        subj_geom = RSA.calculate_geometry(samp_subj)
        obj_geom = RSA.calculate_geometry(samp_obj)
        verb_geom = RSA.calculate_geometry(samp_verb)
        rand_verb_geom = RSA.calculate_geometry(samp_rand_verb)
        adj1_geom = RSA.calculate_geometry(samp_adj1)
        adj2_geom = RSA.calculate_geometry(samp_adj2)

        rsa_subj_dist.append(spearmanr([subj_geom, bert_geom], axis=1)[0])
        rsa_obj_dist.append(spearmanr([obj_geom, bert_geom], axis=1)[0])
        rsa_verb_dist.append(spearmanr([verb_geom, bert_geom], axis=1)[0])
        rsa_adj1_dist.append(spearmanr([adj1_geom, bert_geom], axis=1)[0])
        rsa_adj2_dist.append(spearmanr([adj2_geom, bert_geom], axis=1)[0])
        rsa_rand_verb_dist.append(spearmanr([rand_verb_geom, bert_geom], axis=1)[0])

    # Run Tests
    print(f'RSA Subj: {np.mean(rsa_subj_dist)} STD: {np.std(rsa_subj_dist)}')
    print(f'RSA Obj: {np.mean(rsa_obj_dist)} STD: {np.std(rsa_obj_dist)}')
    print(f'RSA Verb: {np.mean(rsa_verb_dist)} STD: {np.std(rsa_verb_dist)}')
    print(f'RSA Adj: {np.mean(rsa_adj1_dist)} STD: {np.std(rsa_adj1_dist)}')
    print(f'RSA Adj2: {np.mean(rsa_adj2_dist)} STD: {np.std(rsa_adj2_dist)}')
    print(f'RSA Random Verb: {np.mean(rsa_rand_verb_dist)} STD: {np.std(rsa_rand_verb_dist)}')

    print(f'Sign Test Subj vs. Obj: {sign_test(np.array(rsa_subj_dist) - np.array(rsa_obj_dist))[1]}')
    print(f'Sign Test Subj vs. Verb: {sign_test(np.array(rsa_subj_dist) - np.array(rsa_verb_dist))[1]}')
    print(f'Sign Test Verb vs. Obj: {sign_test(np.array(rsa_obj_dist) - np.array(rsa_verb_dist))[1]}')
    print(f'Sign Test Subj vs. Adj: {sign_test(np.array(rsa_subj_dist) - np.array(rsa_adj1_dist))[1]}')
    print(f'Sign Test Obj vs. Adj: {sign_test(np.array(rsa_obj_dist) - np.array(rsa_adj1_dist))[1]}')
    print(f'Sign Test Verb vs. Adj: {sign_test(np.array(rsa_verb_dist) - np.array(rsa_adj1_dist))[1]}')

    print(f'Sign Test Adj vs. Adj2: {sign_test(np.array(rsa_adj1_dist) - np.array(rsa_adj2_dist))[1]}')
    print(f'Sign Test Subj vs. Adj2: {sign_test(np.array(rsa_subj_dist) - np.array(rsa_adj2_dist))[1]}')
    print(f'Sign Test Obj vs. Adj2: {sign_test(np.array(rsa_obj_dist) - np.array(rsa_adj2_dist))[1]}')
    print(f'Sign Test Verb vs. Adj2: {sign_test(np.array(rsa_verb_dist) - np.array(rsa_adj2_dist))[1]}')

    print(f'Sign Test Verb vs. Random Verb: {sign_test(np.array(rsa_rand_verb_dist) - np.array(rsa_verb_dist))[1]}')
