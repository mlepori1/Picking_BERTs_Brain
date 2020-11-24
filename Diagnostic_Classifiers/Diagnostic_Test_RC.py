import torch
from pytorch_pretrained_bert import BertTokenizer, BertModel
import logging
import matplotlib.pyplot as plt
import sys
import numpy as np
sys.path.append("..")
import glove_utils.utils as utils
import RSA_utils.utils as RSA
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.linear_model import LogisticRegression
import random
import matplotlib.pyplot as plt

noun_idxs = [1, 5]
verb_idx = 6

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
            'rabbit',
            'duck',
            'dog',
            'cat',
            'rat',
            'frog',
            'rabbits',
            'ducks',
            'dogs',
            'cats',
            'rats',
            'frogs'
            ]


def get_diagnostic_input(sent_list, glove, word2idx, bert_embeds, ):

    glove_embeds_subj_id = []
    glove_embeds_nonarg_id = []

    for i in range(len(sent_list)):

        sent = sent_list[i]
        bert_pronoun = bert_embeds[i]

        subj = sent[1]
        nonarg = sent[5]

        for word in sent:

            if word == sent[-2]: # Skips copula
                continue

            if word == subj:
                cat_embeds = np.concatenate((bert_pronoun, glove[word2idx[word]]))
                glove_embeds_subj_id.append([cat_embeds, 1])
                glove_embeds_nonarg_id.append([cat_embeds, 0])
            if word == nonarg:
                cat_embeds = np.concatenate((bert_pronoun, glove[word2idx[word]]))
                glove_embeds_subj_id.append([cat_embeds, 0])
                glove_embeds_nonarg_id.append([cat_embeds, 1])
            else:
                cat_embeds = np.concatenate((bert_pronoun, glove[word2idx[word]]))
                glove_embeds_subj_id.append([cat_embeds, 0])
                glove_embeds_nonarg_id.append([cat_embeds, 0])

    return np.array(glove_embeds_nonarg_id), np.array(glove_embeds_subj_id)


if __name__ == "__main__":

    np.random.seed(seed=9)
    random.seed(9)

    glove_list, bert_list = RSA.preprocess_data('../Subject_Tracking/Relative_Clauses/copula_RC_corpus.txt', noun_list)
    print("data processed")

    print("glove embeds generated")
    bert_embeds = RSA.get_bert_embeds(bert_list, verb_idx + 1)
    print("BERT embeds generated")

    word2idx, idx2word = utils.create_word_idx_matrices(glove_list)
    print("word idx matrices created")
    glove = utils.create_embedding_dictionary("../glove_utils/glove/glove.6B.300d.txt", 300, word2idx, idx2word)
    print("glove matrices created")
    
    nonarg_data, subj_data = get_diagnostic_input(glove_list, glove, word2idx, bert_embeds)
    print("glove embeds generated")

    np.random.seed(seed=9)
    np.random.shuffle(nonarg_data)
    np.random.shuffle(subj_data)

    subj_X = np.concatenate(np.array(subj_data)[:, 0]).reshape(-1, 1068)
    subj_Y = np.array(subj_data)[:, 1].reshape(-1).astype("int")
    nonarg_X = np.concatenate(np.array(nonarg_data)[:, 0]).reshape(-1, 1068)
    nonarg_Y = np.array(nonarg_data)[:, 1].reshape(-1).astype("int")

    subj_train_x, subj_test_x, subj_train_y, subj_test_y = train_test_split(subj_X, subj_Y, test_size=.2)
    nonarg_train_x, nonarg_test_x, nonarg_train_y, nonarg_test_y = train_test_split(nonarg_X, nonarg_Y, test_size=.2)

    subj_clf = LogisticRegression(max_iter=5000)
    nonarg_clf = LogisticRegression(max_iter=5000)

    subj_clf.fit(subj_train_x, subj_train_y)
    nonarg_clf.fit(nonarg_train_x, nonarg_train_y)

    subj_preds = subj_clf.predict(subj_test_x)
    nonarg_preds = nonarg_clf.predict(nonarg_test_x)

    print(f'Subject Accuracy: {accuracy_score(subj_test_y, subj_preds)}\nSubject Precision: {precision_score(subj_test_y, subj_preds)}\nSubject Recall: {recall_score(subj_test_y, subj_preds)}')
    print(f'Non-Argument Accuracy: {accuracy_score(nonarg_test_y, nonarg_preds)}\nNon-Argument Precision: {precision_score(nonarg_test_y, nonarg_preds)}\nNon-Argument Recall: {recall_score(nonarg_test_y, nonarg_preds)}')

