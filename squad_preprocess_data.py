import json
from spacy import tokens
from tqdm import tqdm
import re

import spacy
from nltk.tokenize import word_tokenize

from config import Config

def isEnglish(s):
	try:
		s.encode(encoding='utf-8').decode('ascii')
	except UnicodeDecodeError:
		return False
	else:
		return True

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
	phrase = re.sub(r"b'day", "birthday", phrase)
	return phrase

def preprocess_text (text, tokenizer, is_answer=False):
	# For hyphenated tokens
	# text = text.replace ('—', ' ')
	# text = text.replace ('-', ' ')
	# text = text.replace ('–', ' ')
	# text = text.replace ('_', ' ')
	# text = text.replace ("'", " ")

	# For english contractions
	text = decontract (text.lower ())
	# text = text.replace ("’s", ' us')
	# text = text.replace ("'s", ' us')
	# text = text.replace ("’re", ' are')
	# text = text.replace ("'re", ' are')
	# text = text.replace ("’ll", ' will')
	# text = text.replace ("'ll", ' will')
		
	# For tokenization
	# text = text.replace ('.', ' . ')
	# text = text.replace (',', ' , ')
	# text = text.replace ('?', ' ? ')
	# text = text.replace ('!', ' ! ')
	# text = text.replace ('$', ' $ ')
	# text = text.replace ('£', ' £ ')

	tokens = tokenizer (text)
	# tokens = word_tokenize (text)
	allowed_punc = set (['£', '$', ',', '.', '?', '!'])
	filtered_tokens = list ()
		
	for tok in tokens:
		# tok = tok.strip ()
		if tok.text.isalpha () or tok.text.isnumeric ():
			filtered_tokens.append (tok.text)
		elif tok.text in allowed_punc:
			filtered_tokens.append (tok.text)
		# if tok.isalnum () or tok.isnumeric ():
		#	 filtered_tokens.append (tok)
		# elif tok in allowed_punc:
		#	 filtered_tokens.append (tok)
		elif is_answer:
			filtered_tokens.append (tok.text)

	if len (filtered_tokens) == 0:
		return None
	return ' '.join (filtered_tokens)

def preprocess_vqg_corpus (corpus, tokenizer):
	for question_obj in tqdm (corpus):
		question_obj ['question'] = preprocess_text (question_obj ['question'], tokenizer)
		question_obj ['context'] = preprocess_text (question_obj ['context'], tokenizer)
		question_obj ['answer'] = preprocess_text (question_obj ['answer'], tokenizer, is_answer=True)

		if question_obj ['question'] == None or question_obj ['context'] == None or question_obj ['answer'] == None:
			return question_obj ['question_id'], None 
	return -1, corpus

def preprocess_squad_corpus (corpus, allowed_titles, tokenizer):
	prep_obj = list ()
	count = 0

	for topic in corpus ['data']:
		if allowed_titles == None or topic ['title'] in allowed_titles:
			for par in tqdm (topic ['paragraphs'], desc=f"{topic ['title']}"):
				context = preprocess_text (par ['context'], tokenizer)
				if context == None:
					return f"context - {par ['context']}", None 

				for ques_obj in par ['qas']:
					if ques_obj ['is_impossible'] == False:
						question_str = preprocess_text (ques_obj ['question'], tokenizer)

						if len (ques_obj ['answers']) > 0:
							answer = preprocess_text (ques_obj ['answers'] [0] ['text'], tokenizer, is_answer=True)
						elif len (ques_obj ['plausible_answers']) > 0:
							answer = preprocess_text (ques_obj ['plausible_answers'] [0] ['text'], tokenizer, is_answer=True)
						else:
							print (f'{ques_obj}')
							return 'key erro', None
						
						if question_str == None:
							return f"question - {ques_obj ['question']}", None
						if answer == None:
							if len (ques_obj ['answers']) > 0:
								return f"answer - {ques_obj ['answers'] [0] ['text']}", None
							else:
								return f"answer - {ques_obj ['plausible_answers'] [0] ['text']}", None
						
						if isEnglish (question_str) and isEnglish (answer) and isEnglish (context):
						   count += 1
						   prep_obj.append ({
								'question' : question_str,
								'answer' : answer,
								'context' : context
						   })

	return -1, prep_obj, count

if __name__ == '__main__':
	config = Config ()

	with open (config.squad_train_file, 'r', encoding="utf8") as file_io:
		 train = json.load (file_io)
		
	with open (config.squad_val_file, 'r', encoding="utf8") as file_io:
		 val = json.load (file_io)
		
	with open (config.salient_text_file, 'r', encoding="utf8") as file_io:
		 salient_text_list = json.load (file_io)

	allowed_titles = set (['Computational_complexity_theory', 'Packet_switching', 'Private_school', 'Harvard_University', 'University_of_Chicago'])

	tokenizer = spacy.load ('en_core_web_sm', disable=['tok2vec', 'tagger', 'parser', 'ner', 'attribute_ruler', 'lemmatizer'])
	print(tokenizer.pipe_names)

	train_failed_id, prep_train, train_count = preprocess_squad_corpus (train, None, tokenizer)
	if prep_train:
		 with open (config.squad_prep_train_file, 'w') as file_io:
			 json.dump (prep_train, file_io)
	else:
		 print (f'Error : Preprocessing train - {train_failed_id} returned None.')

	val_failed_id, prep_val, val_count = preprocess_squad_corpus (val, None, tokenizer)

	if prep_val:
		 with open (config.squad_prep_val_file, 'w') as file_io:
			 json.dump (prep_val, file_io)
	else:
		 print (f'Error : Preprocessing val - {val_failed_id} returned None.')
		
	failed_id, preprocessed_text_list = preprocess_vqg_corpus (salient_text_list, tokenizer)

	if preprocessed_text_list:
		 with open (config.preprocessed_text_file, 'w') as file_io:
			 json.dump (preprocessed_text_list, file_io)
	else:
		 print (f'Error : Preprocessing {failed_id} returned None.')

	print (f'Squad Train - {train_count}, Squad Val - {val_count}')
	print ('Done !') 


