import os
import pickle
import numpy as np


def get_embedding(word_to_id):
    filename = '../glove.840B.300d.txt'
    dim = 300

    embeddings = np.random.uniform(-0.2, 0.2, (len(word_to_id), dim))
    with open(filename, encoding='utf-8') as fp:
        for line in fp:
            elements = line.strip().split()
            word = elements[0]  # word
            if word in word_to_id:  # If the word is in the vocabulary, use the pre-training result
                try:
                    embeddings[word_to_id[word]] = [float(v) for v in elements[1:]]
                except ValueError:
                    pass

    embeddings[0] = np.zeros(dim, dtype='float32')
    return embeddings
