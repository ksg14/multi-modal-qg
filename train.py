from json import encoder
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
from model.decoder import AttnDecoder, Decoder, GenerationHead, ProphetNetCG, AudioDecoder, VideoDecoder

from transformers import ProphetNetTokenizer

from config import Config
from utils.dataset import VQGDataset
from utils.custom_transforms import prepare_sequence, Resize, ToFloatTensor, Normalize, prepare_sequence

import warnings
warnings.filterwarnings('ignore')

def save_model (model, model_path):
	try:
		torch.save(model.state_dict(), model_path)
		print (f'Model saved to {model_path}')
	except Exception:
		print (f'unable to save model {str (Exception)}')
	return

def create_emb_layer (weights_matrix, non_trainable):
	num_embeddings, embedding_dim = weights_matrix.size()
	emb_layer = Embedding(num_embeddings, embedding_dim)
	emb_layer.load_state_dict({'weight': weights_matrix})
	if non_trainable:
		emb_layer.weight.requires_grad = False
	return emb_layer, num_embeddings, embedding_dim

def get_mem_usage (model):
	mem_params = sum([param.nelement()*param.element_size() for param in model.parameters()])
	mem_bufs = sum([buf.nelement()*buf.element_size() for buf in model.buffers()])
	mem_usage = (mem_params + mem_bufs) / (1024 * 1024)
	return mem_usage

def validate (args, config, av_enc_model, text_dec, audio_dec, video_dec, gen_head, criterion, dataloader, device):
	val_loss = 0.0
	# val_bleu = 0.0
	# val_bleu_1 = 0.0
	# val_bleu_2 = 0.0
	# val_bleu_3 = 0.0
	# val_bleu_4 = 0.0
	n_len = len (dataloader)
		
	av_enc_model.eval () 
	text_dec.eval ()
	audio_dec.eval ()
	video_dec.eval ()
	gen_head.eval ()

	with torch.no_grad ():
		with tqdm(dataloader) as tepoch:
			for frames, audio_file, context, question_src, question_tgt, question_id, question_str  in tepoch:
				frames, audio_file, context, question_src, question_tgt = frames.to (device), audio_file, context.to (device), question_src.to (device), question_tgt.to (device)
				
				tepoch.set_description (f'Validating...')

				loss = 0

				if args.logs:
					print (f'frames - {frames.shape}')
					print (f'context - {context.shape}')
					print (f'question src - {question_src.shape}')
					print (f'question tgt - {question_tgt.shape}')

				audio_emb, video_emb = av_enc_model (audio_file [0], frames)

				audio_frames = audio_emb.shape [0]
				padded_audio_emb = F.pad (audio_emb, (0, 0, 0, config.av_max_length-audio_frames))
				video_frames = video_emb.shape [0]
				padded_video_emb = F.pad (video_emb, (0, 0, 0, config.av_max_length-video_frames))
				
				if args.logs:
					print (f'audio emb - {audio_emb.shape}')
					print (f'video emb - {video_emb.shape}')

				text_out, text_last_hidden = text_dec (context, question_src, question_tgt)				

				audio_dec_hidden = audio_dec.init_state (1)
				video_dec_hidden = video_dec.init_state (1)

				for dec_i in range (question_src.shape [2]):
					audio_dec_output, audio_dec_hidden, audio_attn= audio_dec (question_src [0][0][dec_i], audio_frames, padded_audio_emb, audio_dec_hidden)

					video_dec_output, video_dec_hidden, video_attn= video_dec (question_src [0][0][dec_i], video_frames, padded_video_emb, video_dec_hidden)

					if args.logs:
						print(f'audio out - {audio_dec_output.shape}')
						print(f'video out - {video_dec_output.shape}')
						print(f'text out - {text_out [0][dec_i].shape}')
					
					gen_out = gen_head (audio_dec_output, video_dec_output, text_out [0][dec_i])
					
					loss += criterion (gen_out, question_tgt [0][0][dec_i].view (-1))

				if args.logs:
					print (f'loss - {loss.item () / n_len}')
				
				val_loss += loss.item () / question_src.shape [2]

				# question_str_list = question [0].split ()
				# val_bleu_1 += sentence_bleu (question_str_list, pred_words, weights=(1, 0, 0, 0))
				# val_bleu_2 += sentence_bleu (question_str_list, pred_words, weights=(0.5, 0.5, 0, 0))
				# val_bleu_3 += sentence_bleu (question_str_list, pred_words, weights=(0.33, 0.33, 0.33, 0))
				# # val_bleu_4 += sentence_bleu (question_str_list, pred_words)
				# val_bleu += sentence_bleu (question_str_list, pred_words)
				tepoch.set_postfix (val_loss=(val_loss / n_len))
		
		val_loss = val_loss.item () / n_len
		# val_bleu /= n_len
		# val_bleu_1 /= n_len 
		# val_bleu_2 /= n_len
		# val_bleu_3 /= n_len

	# print (f'Val_loss - {round (val_loss, 3)}, Val_bleu - {round (val_bleu, 3)}, Val_bleu_1 - {round (val_bleu_1, 3)}')
	# return val_loss, val_bleu, val_bleu_1, val_bleu_2, val_bleu_3 
	return val_loss

def train (args, config, av_enc_model, text_dec, audio_dec, video_dec, gen_head, criterion, train_dataloader, val_dataloader, av_enc_optimizer, text_dec_optimizer, audio_dec_optimizer, video_dec_optimizer, gen_head_optimizer, device):
	epoch_stats = { 'train' : {'loss' : []}, 'val' : {'loss' : [], 'bleu' : [], 'bleu_1' : [], 'bleu_2' : [], 'bleu_3' : [], 'bleu_4' : []} }
	n_len = len (train_dataloader)
	best_epoch_loss = float ('inf')
	best_epoch = -1

	for epoch in range (args.epochs):
		epoch_stats ['train']['loss'].append (0.0)
		av_enc_model.train ()
		text_dec.eval ()
		audio_dec.train ()
		video_dec.train ()
		gen_head.train ()

		with tqdm(train_dataloader) as tepoch:
			for frames, audio_file, context, question_src, question_tgt, question_id, question_str  in tepoch:
				frames, audio_file, context, question_src, question_tgt = frames.to (device), audio_file, context.to (device), question_src.to (device), question_tgt.to (device)
				
				tepoch.set_description (f'Epoch {epoch}')

				av_enc_optimizer.zero_grad()
				# text_dec_optimizer.zero_grad ()
				audio_dec_optimizer.zero_grad()
				video_dec_optimizer.zero_grad()
				gen_head_optimizer.zero_grad()
				loss = 0

				if args.logs:
					print (f'frames - {frames.shape}')
					print (f'context - {context.shape}')
					print (f'question src - {question_src.shape}')
					print (f'question tgt - {question_tgt.shape}')

				audio_emb, video_emb = av_enc_model (audio_file [0], frames)

				audio_frames = audio_emb.shape [0]
				padded_audio_emb = F.pad (audio_emb, (0, 0, 0, config.av_max_length-audio_frames))
				video_frames = video_emb.shape [0]
				padded_video_emb = F.pad (video_emb, (0, 0, 0, config.av_max_length-video_frames))
				
				if args.logs:
					print (f'audio emb - {audio_emb.shape}')
					print (f'video emb - {video_emb.shape}')

				text_out, text_last_hidden = text_dec (context, question_src, question_tgt)				

				audio_dec_hidden = audio_dec.init_state (1)
				video_dec_hidden = video_dec.init_state (1)

				for dec_i in range (question_src.shape [2]):
					audio_dec_output, audio_dec_hidden, audio_attn= audio_dec (question_src [0][0][dec_i], audio_frames, padded_audio_emb, audio_dec_hidden)

					video_dec_output, video_dec_hidden, video_attn= video_dec (question_src [0][0][dec_i], video_frames, padded_video_emb, video_dec_hidden)

					if args.logs:
						print(f'audio out - {audio_dec_output.shape}')
						print(f'video out - {video_dec_output.shape}')
						print(f'text out - {text_out [0][dec_i].shape}')
					
					gen_out = gen_head (audio_dec_output, video_dec_output, text_out [0][dec_i])
					
					loss += criterion (gen_out, question_tgt [0][0][dec_i].view (-1))

				if args.logs:
					print (f'loss - {loss.item () / n_len}')

				loss.backward()

				av_enc_optimizer.step()
				# text_dec_optimizer.step()
				audio_dec_optimizer.step()
				video_dec_optimizer.step()
				gen_head_optimizer.step()

				with torch.no_grad():
					epoch_stats ['train']['loss'] [-1] += ((loss.item () / question_src.shape [2]) / n_len)
				
				tepoch.set_postfix (train_loss=epoch_stats ['train']['loss'] [-1])
				# break
		# break
		val_loss = validate (args, config, av_enc_model, text_dec, audio_dec, video_dec, gen_head, criterion, val_dataloader, device)
		epoch_stats ['val']['loss'].append (val_loss)
		# epoch_stats ['val']['bleu'].append (val_bleu)
		# epoch_stats ['val']['bleu_1'].append (val_bleu_1)
		# epoch_stats ['val']['bleu_2'].append (val_bleu_2)
		# epoch_stats ['val']['bleu_3'].append (val_bleu_3)
		# # epoch_stats ['val']['bleu_4'].append (val_bleu_4)

		# Save best model
		if val_loss < best_epoch_loss:
			best_epoch_loss = val_loss
			best_epoch = epoch

			print ('Saving new best model !')
			save_model (av_enc_model, config.av_model_path)
			save_model (audio_dec, config.audio_model_path)
			save_model (video_dec, config.video_model_path)
			save_model (gen_head, config.gen_head_model_path)
			# text_dec.save_model (config.text_model_path)
		
		# Save last epoch model
		if epoch == args.epochs-1:
			print ('Saving last epoch model !')
			save_model (av_enc_model, config.output_path / 'last_av_model.pth')
			save_model (audio_dec, config.output_path / 'last_audio.pth')
			save_model (video_dec, config.output_path / 'last_video.pth')
			save_model (gen_head, config.output_path / 'last_gen_head.pth')
			# text_dec.save_model (config.last_text_model_path)

	return epoch_stats, best_epoch

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Training code')
	parser.add_argument('-l',
						'--logs',
						action='store_true',
						help='print logs')
	parser.add_argument('--epochs', type=int, default=20)
	parser.add_argument('--batch_sz', type=int, default=1)
	parser.add_argument('--lr', type=float, default=1e-4)
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

	av_enc_model = AudioVideoEncoder (config.av_in_channels, config.av_kernel_sz, config.av_stride, config.video_hidden_dim, config.flatten_dim, config.audio_emb, device)

	text_dec = ProphetNetCG (config.pretrained_cg_dec_path)

	# print (f'bos - {text_dec.model.config.bos_token_id}')

	emb_layer = text_dec.model.get_input_embeddings ()

	audio_dec = AudioDecoder (num_layers=config.audio_dec_layers, dropout_p=config.audio_dec_dropout, hidden_dim=config.audio_dec_hidden, n_vocab=config.prophetnet_vocab, word_emb_dim=config.prophetnet_hidden_sz, audio_emb_dim=config.audio_emb, emb_layer=emb_layer, av_max_length=config.av_max_length, device=device)

	video_dec = VideoDecoder (num_layers=config.video_dec_layers, dropout_p=config.video_dec_dropout, hidden_dim=config.video_dec_hidden, n_vocab=config.prophetnet_vocab, word_emb_dim=config.prophetnet_hidden_sz, video_emb_dim=config.video_hidden_dim, emb_layer=emb_layer, av_max_length=config.av_max_length, device=device)

	gen_head = GenerationHead (enc_emb_dim=config.prophetnet_vocab*3, n_vocab=config.prophetnet_vocab, device=device)

	av_enc_model.to (device)
	text_dec.to (device)
	audio_dec.to (device)
	video_dec.to (device)
	gen_head.to (device)

	criterion = CrossEntropyLoss()

	av_enc_optimizer = Adam(av_enc_model.parameters(), lr=args.lr)
	# text_dec_optimizer = Adam(text_dec.parameters(), lr=args.lr)
	audio_dec_optimizer = Adam(audio_dec.parameters(), lr=args.lr)
	video_dec_optimizer = Adam(video_dec.parameters(), lr=args.lr)
	gen_head_optimizer = Adam(gen_head.parameters(), lr=args.lr)

	epoch_stats, best_epoch = train (args=args, config=config, av_enc_model=av_enc_model, \
									text_dec=text_dec, audio_dec=audio_dec, video_dec=video_dec, \
									gen_head=gen_head, train_dataloader=train_dataloader, \
									val_dataloader=val_dataloader, av_enc_optimizer=av_enc_optimizer, \
									text_dec_optimizer=None, audio_dec_optimizer=audio_dec_optimizer, \
									gen_head_optimizer=gen_head_optimizer, video_dec_optimizer=video_dec_optimizer, \
									criterion=criterion, device=device)

	# validate (args, config, av_enc_model, text_enc_model, dec_model, val_dataloader, device)
		
	print (f'Best epoch - {best_epoch} !')

	try:
		with open (config.stats_json_path, 'w') as f:
			json.dump (epoch_stats, f)
			print (f'Stats saved to {config.stats_json_path}')
	except Exception:
		pickle.dump(epoch_stats, open(config.stats_pkl_path, 'wb'))
		print (f'Stats saved to {config.stats_pkl_path}')
		
	try:
		config.save_config ()
	except Exception as e:
		print (f'Unable to save config {str (e)}')
		
	print ('Done !')

	# print (f'mem av - {get_mem_usage (av_enc_model)} MB')
	# print (f'mem text enc - {get_mem_usage (text_enc_model)} MB')
	# print (f'mem dec - {get_mem_usage (dec_model)} MB')
