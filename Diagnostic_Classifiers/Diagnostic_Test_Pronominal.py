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

noun_idxs = [5, 1]
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


def get_diagnostic_input(sent_list, glove, word2idx, bert_embeds, anaphor=True):

    glove_embeds_nonant_id = []
    glove_embeds_ant_id = []

    for i in range(len(sent_list)):

        sent = sent_list[i]
        bert_pronoun = bert_embeds[i]

        if anaphor:
            ant = sent[5]
            nonant = sent[1]
        else:
            ant = sent[1]
            nonant = sent[5]

        for word in sent:

            if word == sent[-1]: # Skips pronoun
                continue

            if word == ant:
                cat_embeds = np.concatenate((bert_pronoun, glove[word2idx[word]]))
                glove_embeds_ant_id.append([cat_embeds, 1])
                glove_embeds_nonant_id.append([cat_embeds, 0])
            if word == nonant:
                cat_embeds = np.concatenate((bert_pronoun, glove[word2idx[word]]))
                glove_embeds_ant_id.append([cat_embeds, 0])
                glove_embeds_nonant_id.append([cat_embeds, 1])
            else:
                cat_embeds = np.concatenate((bert_pronoun, glove[word2idx[word]]))
                glove_embeds_ant_id.append([cat_embeds, 0])
                glove_embeds_nonant_id.append([cat_embeds, 0])

    return np.array(glove_embeds_nonant_id), np.array(glove_embeds_ant_id)


if __name__ == "__main__":

    np.random.seed(seed=9)
    random.seed(9)

    glove_list, bert_list = RSA.preprocess_data('../Binding_Theory/Pronominal/pronominal_corpus.txt', noun_list)
    print("data processed")

    bert_embeds = RSA.get_bert_embeds(bert_list, pro_idx + 1)
    print("BERT embeds generated")

    word2idx, idx2word = utils.create_word_idx_matrices(glove_list)
    print("word idx matrices created")
    glove = utils.create_embedding_dictionary("../glove_utils/glove/glove.6B.300d.txt", 300, word2idx, idx2word)
    print("glove matrices created")
    
    nonant_data, ant_data = get_diagnostic_input(glove_list, glove, word2idx, bert_embeds, anaphor=False)
    print("glove embeds generated")

    np.random.seed(seed=9)
    np.random.shuffle(nonant_data)
    np.random.shuffle(ant_data)

    ant_X = np.concatenate(np.array(ant_data)[:, 0]).reshape(-1, 1068)
    ant_Y = np.array(ant_data)[:, 1].reshape(-1).astype("int")
    nonant_X = np.concatenate(np.array(nonant_data)[:, 0]).reshape(-1, 1068)
    nonant_Y = np.array(nonant_data)[:, 1].reshape(-1).astype("int")

    ant_train_x, ant_test_x, ant_train_y, ant_test_y = train_test_split(ant_X, ant_Y, test_size=.2)
    nonant_train_x, nonant_test_x, nonant_train_y, nonant_test_y = train_test_split(nonant_X, nonant_Y, test_size=.2)

    ant_clf = LogisticRegression(max_iter=5000)
    nonant_clf = LogisticRegression(max_iter=5000)

    ant_clf.fit(ant_train_x, ant_train_y)
    nonant_clf.fit(nonant_train_x, nonant_train_y)

    ant_preds = ant_clf.predict(ant_test_x)
    nonant_preds = nonant_clf.predict(nonant_test_x)

    print(f'Antecedent Accuracy: {accuracy_score(ant_test_y, ant_preds)}\nAntecedent Precision: {precision_score(ant_test_y, ant_preds)}\nAntecedent Recall: {recall_score(ant_test_y, ant_preds)}')
    print(f'NonAntecedent Accuracy: {accuracy_score(nonant_test_y, nonant_preds)}\nNonAntecedent Precision: {precision_score(nonant_test_y, nonant_preds)}\nNonAntecedent Recall: {recall_score(nonant_test_y, nonant_preds)}')

