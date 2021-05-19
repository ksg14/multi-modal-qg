import json
import pickle
import numpy as np

from sklearn.model_selection import train_test_split

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
    glove_matrix = np.load (config.glove_matrix_file)
    glove_word2idx = pickle.load(open(config.glove_idx_file, 'rb'))

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

    print (f'Weight matrix saved to {config.weights_matrix_file}')
    np.save (config.weights_matrix_file, weights_matrix)
    return

def update_vocab (vocab, index_to_word, vocab_idx, text):
    for tok in text.split ():
        if tok not in vocab:
            vocab [tok] = vocab_idx
            index_to_word [vocab_idx] = tok
            vocab_idx += 1
    return vocab_idx

def build_vocab (corpus):
    vocab = dict ()
    index_to_word = dict ()

    vocab ['<pad>'] = 0
    vocab ['<start>'] = 1
    vocab ['<end>'] = 2
    vocab_idx = 3

    index_to_word [0] = '<pad>'
    index_to_word [1] = '<start>'
    index_to_word [2] = '<end>'

    for entry in corpus:
        try:
            vocab_idx = update_vocab (vocab, index_to_word, vocab_idx, entry ['question'])
            vocab_idx = update_vocab (vocab, index_to_word, vocab_idx, entry ['context'])
            vocab_idx = update_vocab (vocab, index_to_word, vocab_idx, entry ['answer'])
        except:
            print (f"Failed for {entry ['question_id']}")
            return None, None
    
    return vocab, index_to_word

def save_vocab (vocab, path):
    print (f'Saving vocab to {path} ...')
    with open (path, 'w') as f:
        json.dump (vocab, f)
    return

if __name__ == '__main__':
    config = Config ()

    with open (config.preprocessed_text_file, 'r') as file_io:
        preprocessed_text = json.load (file_io)

    vocab, index_to_word = build_vocab (preprocessed_text)

    if vocab:
        print (f'Unique words {len (vocab)}')
        
        save_vocab (vocab, config.vocab_file)
        save_vocab (index_to_word, config.index_to_word_file)

        save_weight_matrix (config, vocab)

        split_data (config, preprocessed_text)

    print ('Done !') 
