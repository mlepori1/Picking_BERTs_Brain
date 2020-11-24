import torch
from pytorch_pretrained_bert import BertTokenizer, BertModel
import logging
import matplotlib.pyplot as plt
import sys
import numpy as np
sys.path.append("../")
from scipy.stats import shapiro
from scipy.stats import spearmanr
from scipy.stats import zscore
from scipy.stats import probplot
import random
import RSA_utils.utils as RSA


# Function to test the normality of BERT embeddings after Z-Normalizing
def test_bert_embeds(bert_sents):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')
    model.eval()

    significant = 0
    total = 0

    means = []
    for sent in bert_sents:

        encoding = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(sent))

        # BERT Sentence ID
        segment_ids = [1] * len(encoding)
        tokens_tensor = torch.tensor([encoding])
        segments_tensor = torch.tensor([segment_ids])

        with torch.no_grad():
            encoded_layers, _ = model(tokens_tensor, segments_tensor)

        for i in range(1, len(sent.split()) - 1):
            encoding = encoded_layers[11].reshape(len(tokens_tensor[0]), -1)[i].reshape(-1)
            means.append(np.mean(encoding.numpy()))
            encoding = encoding.numpy()
            encoding = zscore(encoding)
            #encoding = np.random.choice(encoding, 300) #Uncomment to sample without replacement
            stat, p_val = shapiro(encoding)
            total += 1
            if p_val < .05:
                significant += 1
    return (significant + .0) / total, total, means


# Function to generate QQ plots
def qq_bert(bert_sents, word_idx, corpus):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')
    model.eval()

    sent = bert_sents[0]
    word = sent.split()[word_idx]

    encoding = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(sent))

    # BERT sentence ID Stuff
    segment_ids = [1] * len(encoding)
    tokens_tensor = torch.tensor([encoding])
    segments_tensor = torch.tensor([segment_ids])

    with torch.no_grad():
        encoded_layers, _ = model(tokens_tensor, segments_tensor)

    encoding = encoded_layers[11].reshape(len(tokens_tensor[0]), -1)[-2].reshape(-1)
    encoding = zscore(encoding.numpy())
    print(np.min(encoding))
    print(np.max(encoding))

    probplot(encoding, plot=plt)
    plt.ylim(-12, 12)
    plt.xlim(-5, 5)
    plt.title(f"{corpus} QQ Plot: {word}", fontsize=17)
    plt.xlabel('Theoretical Quantiles', fontsize=17)
    plt.ylabel('Ordered Values', fontsize=17)
    plt.savefig(f"{corpus}_qq_plot_{word}")


if __name__ == "__main__":

    print('Analyze Anaphor Corpus')
    _, bert_list = RSA.preprocess_data('../Binding_Theory/Anaphor/anaphor_corpus.txt')
    print("data processed")
    qq_bert(bert_list, -2, 'Anaphor')
    prop, total, means = test_bert_embeds(bert_list)
    print(f'Percentage non-normal: {prop}')
    print(f'Total embeds in unique contexts: {total}')

    print('Analyze Pronominal Corpus')
    _, bert_list = RSA.preprocess_data('../Binding_Theory/Pronominal/pronominal_corpus.txt')
    print("data processed")
    qq_bert(bert_list, -2, 'Pronominal')
    prop, total, means = test_bert_embeds(bert_list)
    print(f'Percentage non-normal: {prop}')
    print(f'Total embeds in unique contexts: {total}')

    print('Analyze Prepositional Phrase Corpus')
    _, bert_list = RSA.preprocess_data('../Subject_Tracking/Prepositional_Phrases/copula_PP_corpus.txt')
    print("data processed")
    qq_bert(bert_list, -3, 'PP')
    prop, total, means = test_bert_embeds(bert_list)
    print(f'Percentage non-normal: {prop}')
    print(f'Total embeds in unique contexts: {total}')

    print('Analyze Relative Clause Corpus')
    _, bert_list = RSA.preprocess_data('../Subject_Tracking/Relative_Clauses/copula_RC_corpus.txt')
    print("data processed")
    qq_bert(bert_list, -3, "RC")
    prop, total, means = test_bert_embeds(bert_list)
    print(f'Percentage non-normal: {prop}')
    print(f'Total embeds in unique contexts: {total}')







