import numpy as np
import os
import pickle
from tqdm import tqdm

from config import Config

def get_matrix (config, uniq_tokens):
    chars = []
    idx = 0
    char2idx = {}

    glove_matrix = np.zeros((uniq_tokens, config.glove_emb_dim))

    print (f'Chars - {uniq_tokens}')

    with open(config.glove_char_file, 'r') as f:
        for l in tqdm (f):
            line = l.split()
            word = line[0]
            chars.append(word)
            char2idx[word] = idx
            vect = np.array(line[1:]).astype(np.float)
            glove_matrix [idx] = vect
            idx += 1
        
    pickle.dump(chars, open(config.glove_chars_file, 'wb'))
    pickle.dump(char2idx, open(config.glove_char2idx_file, 'wb'))
    np.save (config.glove_char_matrix_file, glove_matrix)
    return

def get_char_emb (config):
    vectors = {}

    print (f'Words - {400000}')
    with open(config.glove_file, 'rb') as f:
        for line in tqdm (f):
            line_split = line.decode().strip().split(" ")
            vec = np.array(line_split[1:], dtype=float)
            word = line_split[0]

            for char in word:
                if ord(char) < 128:
                    if char in vectors:
                        vectors[char] = (vectors[char][0] + vec,
                                        vectors[char][1] + 1)
                    else:
                        vectors[char] = (vec, 1)

    with open(config.glove_char_file, 'w') as f2:
        for word in tqdm (vectors):
            avg_vector = np.round(
                (vectors[word][0] / vectors[word][1]), 6).tolist()
            f2.write(word + " " + " ".join(str(x) for x in avg_vector) + "\n")
    return len (vectors)

if __name__ == '__main__' :
    config = Config ()

    uniq_chars = get_char_emb (config)

    print (f'Uniq characters - {uniq_chars}')

    get_matrix (config, uniq_chars)

