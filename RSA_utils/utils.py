import numpy as np
import glove_utils.utils as utils
from collections import defaultdict
from pytorch_pretrained_bert import BertTokenizer, BertModel
import torch
from scipy.stats import spearmanr
import random

# Adds in BERT tokens, removes sentences whose target words are the same, and removes repeat sentences
def preprocess_data(corpus_path, word_list=None):
    glove_list = []
    bert_list = []
    with open(corpus_path) as f:
        for row in f:
            bert_row = "[CLS] " + row + " [SEP]"
            sent = np.array(row.split())
            if word_list is not None:
                word_idx = np.nonzero(np.isin(sent, word_list))
                word_1 = sent[word_idx[0][0]]
                word_2 = sent[word_idx[0][1]]
                
                if word_1 != word_2 and bert_row not in bert_list: 
                    glove_list.append(list(sent))
                    bert_list.append(bert_row)
            else:
                if bert_row not in bert_list:
                    glove_list.append(list(sent))
                    bert_list.append(bert_row)
    print(len(glove_list))
    return glove_list, bert_list


# Gets glove embeddings for hypothesis models and null models
def get_glove_embeds(sent_list, glove_path, dim, idxs, rand_words, idx_of_interest=None):
    
    word2idx, idx2word = utils.create_word_idx_matrices(sent_list)
    print("word idx matrices created")
    glove = utils.create_embedding_dictionary(glove_path, dim, word2idx, idx2word)
    print("glove matrices created")

    embeds_dict = defaultdict(list)

    for sent in sent_list:
        curr_words = []
        for idx in idxs:
            curr_words.append(sent[idx])
        rand_word_list = list(set(rand_words) - set(curr_words))
        rand_word_list.sort()
        rand_word = random.choice(rand_word_list)
        if idx_of_interest is not None:
            word_of_interest = sent[idx_of_interest]
            embed_of_interest = glove[word2idx[word_of_interest]]

            for idx, word in enumerate(curr_words):
                embeds_dict[idxs[idx]].append(np.concatenate((embed_of_interest, glove[word2idx[word]])))

            embeds_dict[-1].append(np.concatenate((embed_of_interest, glove[word2idx[rand_word]])))

        else:
            for idx, word in enumerate(curr_words):
                embeds_dict[idxs[idx]].append(glove[word2idx[word]])

            embeds_dict[-1].append(glove[word2idx[rand_word]])            

    return embeds_dict

# Gets BERT embeddings for reference model
def get_bert_embeds(bert_sents, bert_idx):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')
    model.eval()

    bert_embeds = []
    for sent in bert_sents:
        encoding = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(sent))
        # Weird BERT sentence ID stuff
        segment_ids = [1] * len(encoding)
        tokens_tensor = torch.tensor([encoding])
        segments_tensor = torch.tensor([segment_ids])
        with torch.no_grad():
            encoded_layers, _ = model(tokens_tensor, segments_tensor)
        embed = encoded_layers[11].reshape(len(tokens_tensor[0]), -1)[bert_idx].reshape(-1)
        bert_embeds.append(embed.numpy())

    return np.array(bert_embeds)

# Calculates the representational geometry of a set of embeddings
def calculate_geometry(sample_embeds):
    sim_mat = spearmanr(sample_embeds, axis=1)[0]
    dissim_mat = np.ones(sim_mat.shape) - sim_mat 
    geometry = dissim_mat[np.triu_indices(sample_embeds.shape[0], 1)].reshape(-1)
    return geometry
