import numpy as np
import pickle
from config import Config

def main (config):
    words = []
    idx = 0
    word2idx = {}

    glove_matrix = np.zeros((400000, config.glove_emb_dim))

    with open(config.glove_file, 'rb') as f:
        for l in f:
            line = l.decode().split()
            word = line[0]
            words.append(word)
            word2idx[word] = idx
            vect = np.array(line[1:]).astype(np.float)
            glove_matrix [idx] = vect
            idx += 1
        
    pickle.dump(words, open(config.glove_words_file, 'wb'))
    pickle.dump(word2idx, open(config.glove_idx_file, 'wb'))
    np.save (config.glove_matrix_file, glove_matrix)

if __name__ == '__main__':
    config = Config ()

    main (config)

