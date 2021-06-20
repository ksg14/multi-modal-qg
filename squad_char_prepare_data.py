import json
import pickle
from squad_preprocess_data import preprocess_text
import numpy as np

from sklearn.model_selection import train_test_split

import spacy

from config import Config

def split_data (config, corpus):
    print ('Generating Splits ...')
    train, rem = train_test_split(corpus, test_size=0.2, random_state=42)
    val, test = train_test_split(rem, test_size=0.5, random_state=42)

    print (f'train - {len (train)}')
    print (f'val - {len (val)}')
    print (f'test - {len (test)}')

    with open (config.train_file, 'w') as file_io:
        json.dump (train, file_io)
    with open (config.val_file, 'w') as file_io:
        json.dump (val, file_io)
    with open (config.test_file, 'w') as file_io:
        json.dump (test, file_io)
    return

def save_weight_matrix (config, wtoi):
    glove_matrix = np.load (config.glove_char_matrix_file)
    glove_word2idx = pickle.load(open(config.glove_char2idx_file, 'rb'))

    matrix_len = len(wtoi.keys ())
    emb_dim = config.glove_emb_dim
    weights_matrix = np.zeros((matrix_len, emb_dim))
    words_found = 0

    for word, idx in wtoi.items ():
        try:
            weights_matrix[idx] = glove_matrix [glove_word2idx [word]]
            words_found += 1
        except KeyError:
            # print (word)
            if word != '<pad>':
                weights_matrix[idx] = np.random.normal(scale=0.6, size=(emb_dim, ))
    
    print (f'Words found : {words_found}')
    print (f'Vocab words : {matrix_len}')

    print (f'Weight matrix saved to {config.char_weights_matrix_file}')
    np.save (config.char_weights_matrix_file, weights_matrix)
    return

def update_vocab (vocab, index_to_char, vocab_idx, text):
    for char in text:
        if char.isspace ():
            continue
        else:
            if char not in vocab:
                vocab [char] = vocab_idx
                index_to_char [vocab_idx] = char
                vocab_idx += 1
    return vocab_idx

def build_vocab (squad_corpus, vqg_corpus):
    vocab = dict ()
    index_to_char = dict ()

    vocab ['<pad>'] = 0
    vocab ['<start>'] = 1
    vocab ['<end>'] = 2
    vocab ['<sep>'] = 3
    vocab ['<unk>'] = 4
    vocab_idx = 5

    index_to_char [0] = '<pad>'
    index_to_char [1] = '<start>'
    index_to_char [2] = '<end>'
    index_to_char [3] = '<sep>'
    index_to_char [4] = '<unk>'

    for entry in vqg_corpus:
        try:
            vocab_idx = update_vocab (vocab, index_to_char, vocab_idx, entry ['question'])
            vocab_idx = update_vocab (vocab, index_to_char, vocab_idx, entry ['context'])
            vocab_idx = update_vocab (vocab, index_to_char, vocab_idx, entry ['answer'])
        except:
            print (f"Failed for vqg {entry ['question_id']}")
            return None, None
    
    for entry in squad_corpus:
        try:
            vocab_idx = update_vocab (vocab, index_to_char, vocab_idx, entry ['question'])
            vocab_idx = update_vocab (vocab, index_to_char, vocab_idx, entry ['context'])
            vocab_idx = update_vocab (vocab, index_to_char, vocab_idx, entry ['answer'])
        except:
            print (f"Failed for squad {entry ['question']}")
            return None, None
    
    return vocab, index_to_char

def save_vocab (vocab, path):
    print (f'Saving vocab to {path} ...')
    with open (path, 'w') as f:
        json.dump (vocab, f)
    return

if __name__ == '__main__':
    config = Config ()

    with open (config.preprocessed_text_file, 'r') as file_io:
        preprocessed_text = json.load (file_io)
    
    with open (config.squad_prep_train_file, 'r', encoding="utf8") as file_io:
        prep_train = json.load (file_io)
    
    with open (config.squad_prep_val_file, 'r', encoding="utf8") as file_io:
        prep_val = json.load (file_io)

    vocab, index_to_char = build_vocab (prep_train + prep_val, preprocess_text)

    if vocab:
        print (f'Unique words {len (vocab)}')
        
        save_vocab (vocab, config.char_vocab_file)
        save_vocab (index_to_char, config.index_to_char_file)

        save_weight_matrix (config, vocab)

        split_data (config, preprocessed_text)

    print ('Done !') 
