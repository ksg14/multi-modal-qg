import json
from tqdm import tqdm
import re

from nltk.tokenize import word_tokenize 

from config import Config

def decontract (phrase):
    # specific
    phrase = re.sub(r"won\'t", "will not", phrase)
    phrase = re.sub(r"can\'t", "can not", phrase)
    phrase = re.sub(r"let\'s", "let us", phrase)
    phrase = re.sub(r"let’s", "let us", phrase)

    # general
    phrase = re.sub(r"n\'t", " not", phrase)
    phrase = re.sub(r"\'re", " are", phrase)
    phrase = re.sub(r"\'s", " is", phrase)
    phrase = re.sub(r"\'d", " would", phrase)
    phrase = re.sub(r"\'ll", " will", phrase)
    phrase = re.sub(r"\'t", " not", phrase)
    phrase = re.sub(r"\'ve", " have", phrase)
    phrase = re.sub(r"\'m", " am", phrase)
    phrase = re.sub(r"n’t", " not", phrase)
    phrase = re.sub(r"’re", " are", phrase)
    phrase = re.sub(r"’s", " is", phrase)
    phrase = re.sub(r"’d", " would", phrase)
    phrase = re.sub(r"’ll", " will", phrase)
    phrase = re.sub(r"’t", " not", phrase)
    phrase = re.sub(r"’ve", " have", phrase)
    phrase = re.sub(r"’m", " am", phrase)
    return phrase

def preprocess_text (text):
    # For hyphenated tokens
    text = text.replace ('—', ' ')
    text = text.replace ('-', ' ')
    text = text.replace ('_', ' ')

    # For english contractions
    text = decontract (text)
    # text = text.replace ("’s", ' us')
    # text = text.replace ("'s", ' us')
    # text = text.replace ("’re", ' are')
    # text = text.replace ("'re", ' are')
    # text = text.replace ("’ll", ' will')
    # text = text.replace ("'ll", ' will')
    
    # For tokenization
    text = text.replace ('.', ' .')
    text = text.replace (',', ' ,')
    text = text.replace ('?', ' ?')
    text = text.replace ('!', ' !')

    tokens = word_tokenize (text)
    allowed_punc = set ([',', '.', '?', '!'])
    filtered_tokens = list ()
    
    for tok in tokens:
        if tok.isalpha () or tok.isnumeric ():
            filtered_tokens.append (tok)
        elif tok in allowed_punc:
            filtered_tokens.append (tok)
    
    if len (filtered_tokens) == 0:
        return None
    return ' '.join (filtered_tokens)

def preprocess_corpus (corpus):
    for question_obj in tqdm (corpus):
        question_obj ['question'] = preprocess_text (question_obj ['question'])
        question_obj ['context'] = preprocess_text (question_obj ['context'])
        question_obj ['answer'] = preprocess_text (question_obj ['answer'])

        if question_obj ['question'] == None or question_obj ['context'] == None or question_obj ['answer'] == None:
            return question_obj ['question_id'], None 
    return -1, corpus

# def get_max_len (corpus):
#     max_len_questions = 0
#     max_len_context = 0
#     pass


if __name__ == '__main__':
    config = Config ()

    with open (config.salient_text_file, 'r', encoding="utf8") as file_io:
        salient_text_list = json.load (file_io)

    failed_id, preprocessed_text_list = preprocess_corpus (salient_text_list)

    if preprocessed_text_list:
        with open (config.preprocessed_text_file, 'w') as file_io:
            json.dump (preprocessed_text_list, file_io)
    else:
        print (f'Error : Preprocessing {failed_id} returned None.')

    print ('Done !') 


