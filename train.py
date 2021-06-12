import torch
from torch.nn import Embedding, CrossEntropyLoss
from torch.nn import functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import transforms as T

from nltk.translate.bleu_score import sentence_bleu

import numpy as np
from tqdm import tqdm
import json
import pickle
import argparse

from model.encoder import AudioVideoEncoder, TextEncoder, ProphetNetTextEncoder
from model.decoder import AttnDecoder, Decoder, ProphetNetDecoder

from transformers import ProphetNetTokenizer

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

def save_weights (emb_layer, weight_path):
	try:
		torch.save(emb_layer.weight, weight_path)
		print (f'Emb Weights saved to {weight_path}')
	except Exception:
		print (f'unable to save weights {str (Exception)}')
	return

def save_model (model, model_path):
	try:
		torch.save(model.state_dict(), model_path)
		print (f'Model saved to {model_path}')
	except Exception:
		print (f'unable to save model {str (Exception)}')
	return

def get_mem_usage (model):
	mem_params = sum([param.nelement()*param.element_size() for param in model.parameters()])
	mem_bufs = sum([buf.nelement()*buf.element_size() for buf in model.buffers()])
	mem_usage = (mem_params + mem_bufs) / (1024 * 1024)
	return mem_usage

def repackage_hidden(h):
	if isinstance(h, torch.Tensor):
		return h.detach()
	else:
		return tuple(repackage_hidden(v) for v in h)

def validate (args, config, av_enc_model, text_enc_model, dec_model, dataloader, device):
	val_loss = 0.0
	val_bleu = 0.0
	val_bleu_1 = 0.0
	val_bleu_2 = 0.0
	val_bleu_3 = 0.0
	# val_bleu_4 = 0.0
	n_len = len (dataloader)
		
	av_enc_model.eval () 
	text_enc_model.eval ()
	dec_model.eval ()

	with torch.no_grad ():
		with tqdm(dataloader) as tepoch:
			for frames, audio_file, context, question_src, question_tgt  in tepoch:
				frames, audio_file, context, question_src, question_tgt = frames.to (device), audio_file, context.to (device), question_src.to (device), question_tgt.to (device)
				
				tepoch.set_description (f'Validating...')

				av_enc_optimizer.zero_grad()
				text_enc_optimizer.zero_grad ()
				dec_optimizer.zero_grad()

				audio_emb, video_emb = av_enc_model (audio_file [0], frames)

				break
				
				# val_loss += loss.item () / target_len

				# question_str_list = question [0].split ()
				# val_bleu_1 += sentence_bleu (question_str_list, pred_words, weights=(1, 0, 0, 0))
				# val_bleu_2 += sentence_bleu (question_str_list, pred_words, weights=(0.5, 0.5, 0, 0))
				# val_bleu_3 += sentence_bleu (question_str_list, pred_words, weights=(0.33, 0.33, 0.33, 0))
				# # val_bleu_4 += sentence_bleu (question_str_list, pred_words)
				# val_bleu += sentence_bleu (question_str_list, pred_words)
				# tepoch.set_postfix (val_loss=val_loss, val_bleu=val_bleu)
		
	# val_loss = val_loss.item () / n_len
	val_bleu /= n_len
	val_bleu_1 /= n_len 
	val_bleu_2 /= n_len
	val_bleu_3 /= n_len

	print (f'Val_loss - {round (val_loss, 3)}, Val_bleu - {round (val_bleu, 3)}, Val_bleu_1 - {round (val_bleu_1, 3)}')
	return val_loss, val_bleu, val_bleu_1, val_bleu_2, val_bleu_3 

def train (args, config, av_enc_model, text_enc_model, dec_model, train_dataloader, val_dataloader, av_enc_optimizer, text_enc_optimizer, dec_optimizer, device):
	epoch_stats = { 'train' : {'loss' : []}, 'val' : {'loss' : [], 'bleu' : [], 'bleu_1' : [], 'bleu_2' : [], 'bleu_3' : [], 'bleu_4' : []} }
	n_len = len (train_dataloader)
	best_epoch_score = float ('inf')
	best_epoch = -1

	for epoch in range (args.epochs):
		epoch_stats ['train']['loss'].append (0.0)
		av_enc_model.train ()
		text_enc_model.train ()
		dec_model.train ()

		with tqdm(train_dataloader) as tepoch:
			for frames, audio_file, context, question_src, question_tgt  in tepoch:
				frames, audio_file, context, question_src, question_tgt = frames.to (device), audio_file, context.to (device), question_src.to (device), question_tgt.to (device)
				
				tepoch.set_description (f'Epoch {epoch}')

				av_enc_optimizer.zero_grad()
				text_enc_optimizer.zero_grad ()
				dec_optimizer.zero_grad()

				audio_emb, video_emb = av_enc_model (audio_file [0], frames)

				loss.backward()

				av_enc_optimizer.step()
				text_enc_optimizer.step ()
				dec_optimizer.step()

				with torch.no_grad():
					epoch_stats ['train']['loss'] [-1] += ((loss.item () / target_len) / n_len).item ()
				
				tepoch.set_postfix (train_loss=epoch_stats ['train']['loss'] [-1])
				# break
		# break
		val_loss, val_bleu, val_bleu_1, val_bleu_2, val_bleu_3 = validate (args, config, av_enc_model, text_enc_model, dec_model, val_dataloader, device)
		epoch_stats ['val']['loss'].append (val_loss)
		epoch_stats ['val']['bleu'].append (val_bleu)
		epoch_stats ['val']['bleu_1'].append (val_bleu_1)
		epoch_stats ['val']['bleu_2'].append (val_bleu_2)
		epoch_stats ['val']['bleu_3'].append (val_bleu_3)
		# epoch_stats ['val']['bleu_4'].append (val_bleu_4)

		# Save best model
		if val_loss < best_epoch_score:
			best_epoch_score = val_loss
			best_epoch = epoch

			print ('Saving new best model !')
			save_model (av_enc_model, config.av_model_path)
			save_model (text_enc_model, config.text_enc_model_path)
			save_model (dec_model, config.dec_model_path)
			# save_weights (dec_model.emb_layer, config.learned_weight_path)
		
		# Save last epoch model
		if epoch == args.epochs-1:
			print ('Saving last epoch model !')
			save_model (av_enc_model, config.output_path / 'last_av_model.pth')
			save_model (text_enc_model, config.output_path / 'last_text_enc.pth')
			save_model (dec_model, config.output_path / 'last_decoder.pth')
			# save_weights (dec_model.emb_layer, config.output_path / 'last_weigths.pt')

		# print({ 'epoch': epoch, 'train_loss': epoch_stats ['train']['loss'] [-1] })
		# break
	return epoch_stats, best_epoch

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Training code')
	parser.add_argument('-l',
						'--logs',
						action='store_true',
						help='print logs')
	parser.add_argument('--epochs', type=int, default=20)
	parser.add_argument('--batch_sz', type=int, default=1)
	parser.add_argument('--lr', type=float, default=1e-6)
	parser.add_argument('--device', type=str, default='cpu')

	args = parser.parse_args()

	config = Config ()

	if torch.cuda.is_available():
		print ('Cuda is available!')
	
	device = torch.device(args.device)
	print(f'Device - {device}')
	
	tokenizer = ProphetNetTokenizer.from_pretrained (config.pretrained_tokenizer_path)
	video_transform = T.Compose ([ToFloatTensor (), Resize (112)])

	train_dataset = VQGDataset (config.train_file, config.vocab_file, config.index_to_word_file, config.salient_frames_path, config.salient_audio_path, text_transform= None, prophetnet_transform=tokenizer, video_transform=video_transform)
	val_dataset = VQGDataset (config.val_file, config.vocab_file, config.index_to_word_file, config.salient_frames_path, config.salient_audio_path, text_transform= None, prophetnet_transform=tokenizer, video_transform=video_transform)
	train_dataloader = DataLoader (train_dataset, batch_size=args.batch_sz, shuffle=True)
	val_dataloader = DataLoader (val_dataset, batch_size=args.batch_sz, shuffle=True)	

	av_enc_model = AudioVideoEncoder (config.av_in_channels, config.av_kernel_sz, config.av_stride, config.video_hidden_dim, config.flatten_dim, device)

	text_enc_model = ProphetNetTextEncoder (config.pretrained_encoder_path)
		
	dec_model = ProphetNetDecoder (config.pretrained_decoder_path)

	av_enc_model.to (device)
	text_enc_model.to (device)
	dec_model.to (device)

	av_enc_optimizer = Adam(av_enc_model.parameters(), lr=args.lr)
	text_enc_optimizer = Adam(text_enc_model.parameters(), lr=args.lr)
	dec_optimizer = Adam(dec_model.parameters(), lr=args.lr)

	# epoch_stats, best_epoch = train (args=args, config=config, av_enc_model=av_enc_model, text_enc_model=text_enc_model, dec_model=dec_model, \
	# 								train_dataloader=train_dataloader, val_dataloader=val_dataloader, \
	# 								av_enc_optimizer=av_enc_optimizer, text_enc_optimizer=text_enc_optimizer, \
	# 								dec_optimizer=dec_optimizer, device=device)

	validate (args, config, av_enc_model, text_enc_model, dec_model, val_dataloader, device)
		
	# print (f'Best epoch - {best_epoch} !')

	# try:
	# 	with open (config.stats_json_path, 'w') as f:
	# 		json.dump (epoch_stats, f)
	# 		print (f'Stats saved to {config.stats_json_path}')
	# except Exception:
	# 	pickle.dump(epoch_stats, open(config.stats_pkl_path, 'wb'))
	# 	print (f'Stats saved to {config.stats_pkl_path}')
		
	# try:
	# 	config.save_config ()
	# except Exception as e:
	# 	print (f'Unable to save config {str (e)}')
		
		
	print ('Done !')

	# print (f'mem av - {get_mem_usage (av_enc_model)} MB')
	# print (f'mem text enc - {get_mem_usage (text_enc_model)} MB')
	# print (f'mem dec - {get_mem_usage (dec_model)} MB')
