from numpy.lib.function_base import _quantile_dispatcher
import torch
from torch.nn import Embedding, CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import transforms as T

from nltk.translate.bleu_score import sentence_bleu

import numpy as np
from tqdm import tqdm
import json
import pickle

from model.encoder import AudioVideoEncoder, TextEncoder
from model.decoder import AttnDecoder, Decoder

from config import Config
from utils.dataset import VQGDataset
from utils.custom_transforms import prepare_sequence, Resize, ToFloatTensor, Normalize, prepare_sequence

import warnings
warnings.filterwarnings('ignore')

def create_emb_layer (weights_matrix, non_trainable):
	num_embeddings, embedding_dim = weights_matrix.size()
	emb_layer = Embedding(num_embeddings, embedding_dim)
	emb_layer.load_state_dict({'weight': weights_matrix})
	if non_trainable:
		emb_layer.weight.requires_grad = False
	return emb_layer, num_embeddings, embedding_dim

def evaluate (av_enc_model, text_enc_model, dec_model, dataloader, context_max_len, pred_max_len, device):
	# val_loss = 0.0
	val_bleu = 0.0
	val_bleu_1 = 0.0
	val_bleu_2 = 0.0
	val_bleu_3 = 0.0
	val_bleu_4 = 0.0
	n_len = len (dataloader)

	predictions = []

	with torch.no_grad ():
		with tqdm(dataloader) as tepoch:
			for frames, audio_file, context_tensor, question_id, question, target, context_len, target_len in tepoch:
				frames, audio_file, context_tensor, question_id, question, target, context_len, target_len = frames.to (device), audio_file, context_tensor.to (device), question_id, question, target.to (device), context_len.to (device), target_len.to (device)
				
				tepoch.set_description (f'Validating ...')

				av_enc_out = av_enc_model (audio_file [0], frames)

				text_enc_hidden = text_enc_model.init_state (1)
				all_enc_outputs = torch.zeros(context_max_len, text_enc_model.hidden_dim).to (device)

				for ei in range (context_len):
					enc_output, text_enc_hidden = text_enc_model(context_tensor [0][ei], text_enc_hidden)
					all_enc_outputs [ei] = enc_output [0, 0]

				# loss = 0
				dec_input = torch.tensor([[dataloader.dataset.vocab ['<start>']]]).to (device)
				dec_hidden = text_enc_hidden

				pred_words = []

				for di in range(pred_max_len):
					dec_output, dec_hidden, dec_attention = dec_model (dec_input, context_len, av_enc_out, dec_hidden, all_enc_outputs)
					# loss += criterion (dec_output, target [0][di].view (-1))
					
					# Sampling
					# last_word_logits = y_pred[0][-1]
					# p = torch.nn.functional.softmax(last_word_logits, dim=0).detach().numpy()
					# word_index = np.random.choice(len(last_word_logits), p=p)
					# pred_words.append(dataloader.dataset.index_to_word [str (word_index)])
					
					# topk
					topv, topi = dec_output.data.topk(1)
					pred_words.append(dataloader.dataset.index_to_word [str (topi.item ())])

					dec_input = topi.squeeze().detach().to (device)

					if pred_words [-1] == '<end>':
						del pred_words [-1]
						break
				# break

				# val_loss += (loss.item () / target_len)
				question_str_list = question [0].split ()
				val_bleu_1 += sentence_bleu (question_str_list, pred_words, weights=(1, 0, 0, 0))
				val_bleu_2 += sentence_bleu (question_str_list, pred_words, weights=(0.5, 0.5, 0, 0))
				val_bleu_3 += sentence_bleu (question_str_list, pred_words, weights=(0.33, 0.33, 0.33, 0))
				val_bleu_4 += sentence_bleu (question_str_list, pred_words)
				val_bleu += sentence_bleu (question_str_list, pred_words)
				
				predictions.append ({
					'question_id' : question_id [0].item (),
					'gt_question' : question [0],
					'pred_question' : ' '.join (pred_words)
				})

				tepoch.set_postfix (val_bleu=val_bleu)

	print (f'Val_bleu - {round (val_bleu, 3)}, Val_bleu_1 - {round (val_bleu_1, 3)}')
	return predictions, val_bleu / n_len, val_bleu_1 / n_len, val_bleu_2 / n_len, val_bleu_3 / n_len, val_bleu_4 / n_len 

if __name__ == '__main__':
	config = Config ()

	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	print(f'Device - {device}')

	av_emb = 128 + 400 # + 128
	question_max_length = 21
	context_max_lenth = 283

	# weights_matrix = torch.from_numpy(np.load (config.weights_matrix_file))
	# weights_matrix = weights_matrix.long ().to (device)

	mean = [0.43216, 0.394666, 0.37645]
	std = [0.22803, 0.22145, 0.216989]
	video_transform = T.Compose ([ToFloatTensor (), Resize (112), Normalize (mean, std)])

	test_dataset = VQGDataset (config.test_file, config.vocab_file, config.index_to_word_file, config.salient_frames_path, config.salient_audio_path, text_transform= prepare_sequence, video_transform=video_transform)
	test_dataloader = DataLoader (test_dataset, batch_size=1, shuffle=False)

	weights_matrix = torch.load (config.learned_weight_path)
	emb_layer, n_vocab, emb_dim = create_emb_layer (weights_matrix, True)	

	av_enc_model = AudioVideoEncoder ()
	av_enc_model.eval ()

	text_enc_model = TextEncoder (num_layers=config.text_lstm_layers, \
									dropout=config.text_lstm_dropout, \
									hidden_dim=config.text_lstm_hidden_dim, \
									emb_dim=emb_dim, \
									emb_layer=emb_layer, \
									device=device)
	text_enc_model.load_state_dict(torch.load(config.text_enc_model_path))
	text_enc_model.eval()

	dec_model = AttnDecoder (num_layers=config.dec_lstm_layers, \
								dropout_p=config.dec_lstm_dropout, \
								hidden_dim=config.dec_lstm_hidden_dim, \
								n_vocab=n_vocab, \
								word_emb_dim=emb_dim, \
								av_emb_dim=av_emb, \
								emb_layer=emb_layer, \
								max_length=context_max_lenth, \
								device=device)
	dec_model.load_state_dict(torch.load(config.dec_model_path))
	dec_model.eval ()

	av_enc_model.to (device)
	text_enc_model.to (device)
	dec_model.to (device)

	predictions, val_bleu, val_bleu_1, val_bleu_2, val_bleu_3, val_bleu_4  = evaluate (av_enc_model, text_enc_model, dec_model, test_dataloader, context_max_lenth, question_max_length, device)

	with open (config.predictions_json_path, 'w') as file_io:
		json.dump (predictions, file_io)
		print (f'Predictions saved to {config.predictions_json_path}')	

	print ('Done !')

