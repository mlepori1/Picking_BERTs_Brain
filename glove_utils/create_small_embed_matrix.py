


import pickle

word2idx = pickle.load(open('./data/word2idx.pkl', 'rb'))
idx2word = {}
for key, val in word2idx.items():
    idx2word[val] = key

glove_embeds = utils.create_embedding_dictionary('./data/glove/glove.6B.50d.txt', 50, word2idx, idx2word)
print('Done Processing Word Vectors')

pickle.dump(glove_embeds, open('./data/small_embed_matrix.pkl', 'wb'))