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
from model.decoder import AttnDecoder, Decoder, ProphetNetCG, AudioDecoder, VideoDecoder, GenerationHead

from transformers import ProphetNetTokenizer

from config import Config
from utils.dataset import VQGDataset
from utils.custom_transforms import prepare_sequence, Resize, ToFloatTensor, Normalize, prepare_sequence

import warnings
warnings.filterwarnings('ignore')


def evaluate (args, config, tokenizer, av_enc_model, text_dec, audio_dec, video_dec, gen_head, dataloader, device):
	# val_bleu = 0.0
	# val_bleu_1 = 0.0
	# val_bleu_2 = 0.0
	# val_bleu_3 = 0.0
	# val_bleu_4 = 0.0
	n_len = len (dataloader)
	predictions = []
	pred_ids = []

	av_enc_model.eval () 
	text_dec.eval ()
	audio_dec.eval ()
	video_dec.eval ()
	gen_head.eval ()

	with torch.no_grad ():
		with tqdm(dataloader) as tepoch:
			for frames, audio_file, context, question_src, question_tgt, question_id, question_str  in tepoch:
				frames, audio_file, context, question_src, question_tgt = frames.to (device), audio_file, context.to (device), question_src.to (device), question_tgt.to (device)
				
				tepoch.set_description (f'Evaluating...')

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
					print (f'audio emb - {padded_audio_emb.shape}')
					print (f'video emb - {padded_video_emb.shape}')
					# print (f'enc hidden - {enc_hidden_state.shape}')

				# enc_out = torch.cat ([enc_hidden_state, audio_emb.unsqueeze (0), video_emb.unsqueeze (0)], dim=1)

				# if args.logs:
					# print (f'enc out - {enc_out.shape}')

				text_out = text_dec.generate (context=context, strategy=args.strategy, beams=args.beams, max_len=args.max_len)
				
				text_out_len = len (text_out.scores)

				if args.logs:
					print (f'text scores - {len (text_out.scores)}')
				
				audio_dec_hidden = audio_dec.init_state (1)
				video_dec_hidden = video_dec.init_state (1)

				dec_input = torch.tensor([[tokenizer.encode ('[CLS]')]]).to (device)

				for dec_i in range (text_out_len):
					audio_dec_output, audio_dec_hidden, audio_attn= audio_dec (dec_input, audio_frames, padded_audio_emb, audio_dec_hidden)

					video_dec_output, video_dec_hidden, video_attn= video_dec (dec_input, video_frames, padded_video_emb, video_dec_hidden)

					if args.logs:
						print(f'audio out - {audio_dec_output.shape}')
						print(f'video out - {video_dec_output.shape}')
						print(f'text out - {text_out.scores [dec_i].shape}')
					
					gen_out = gen_head (audio_dec_output, video_dec_output, text_out.scores [dec_i])

					next_word = torch.argmax (F.log_softmax (gen_out), dim=1)

					if args.logs:
						print (f'next word - {next_word}')

					dec_input = next_word.unsqueeze (0).detach ()

					pred_ids.append (next_word.item ())

				if args.logs:
					print (f'pred ids {pred_ids}')
				# pred_question_str = tokenizer.decode(pred_question_ids [0], skip_special_tokens=True)

				# audio_dec_hidden = audio_dec.init_state (1)
				# video_dec_hidden = video_dec.init_state (1)

				# for dec_i in range (question_src.shape [2]):
				# 	audio_dec_output, audio_dec_hidden, audio_attn= audio_dec (question_src [0][0][dec_i], audio_frames, padded_audio_emb, audio_dec_hidden)

				# 	video_dec_output, video_dec_hidden, video_attn= video_dec (question_src [0][0][dec_i], video_frames, padded_video_emb, video_dec_hidden)

				# 	if args.logs:
				# 		print(f'audio out - {audio_dec_output.shape}')
				# 		print(f'video out - {video_dec_output.shape}')
				# 		print(f'text out - {text_out [0][dec_i].shape}')
					
				# 	gen_out = gen_head (audio_dec_output, video_dec_output, text_out [0][dec_i])

				# predictions.append ({
				# 	'question_id' : question_id [0].item (),
				# 	'gt_question' : question_str [0], 
				# 	'pred_question' : pred_question_str
				# })


				# question_str_list = question [0].split ()
				# val_bleu_1 += sentence_bleu (question_str_list, pred_words, weights=(1, 0, 0, 0))
				# val_bleu_2 += sentence_bleu (question_str_list, pred_words, weights=(0.5, 0.5, 0, 0))
				# val_bleu_3 += sentence_bleu (question_str_list, pred_words, weights=(0.33, 0.33, 0.33, 0))
				# # val_bleu_4 += sentence_bleu (question_str_list, pred_words)
				# val_bleu += sentence_bleu (question_str_list, pred_words)
				# tepoch.set_postfix (val_loss=(val_loss / n_len))
		
		# val_bleu /= n_len
		# val_bleu_1 /= n_len 
		# val_bleu_2 /= n_len
		# val_bleu_3 /= n_len

	# print (f'Val_loss - {round (val_loss, 3)}, Val_bleu - {round (val_bleu, 3)}, Val_bleu_1 - {round (val_bleu_1, 3)}')
	# return val_loss, val_bleu, val_bleu_1, val_bleu_2, val_bleu_3 
	return predictions 

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Evaluate model')
	parser.add_argument('-b',
						'--best',
						action='store_true',
						help='get best epoch results')
	parser.add_argument('-l',
						'--last',
						action='store_true',
						help='get last epoch results')
	parser.add_argument('-p',
						'--pretrained',
						action='store_true',
						help='get pretrained model results')
	parser.add_argument('--logs',
						action='store_true',
						help='get logs')
	parser.add_argument('-c', 
						'--config_path',
						type=str,
						required=True)
	parser.add_argument('-s', 
						'--strategy', # greedy, beam
						type=str,
						required=True)
	parser.add_argument('--batch_sz', type=int, default=1)
	parser.add_argument('--max_len', type=int, default=21)
	parser.add_argument('--beams', type=int, default=5)
	parser.add_argument('--device', type=str, default='cpu')

	args = parser.parse_args()

	# try:
	# 	config = Config (args.config_path)
	# except Exception as e:
	# 	print (f' Config load error {str (e)}')
	# 	config = None

	config= Config ()

	if config:
		device = torch.device(args.device)
		print(f'Device - {device}')

		tokenizer = ProphetNetTokenizer.from_pretrained (config.pretrained_tokenizer_path)
		video_transform = T.Compose ([ToFloatTensor (), Resize (112)])
		
		test_dataset = VQGDataset (config.test_file, config.vocab_file, config.index_to_word_file, config.salient_frames_path, config.salient_audio_path, text_transform= None, prophetnet_transform=tokenizer, video_transform=video_transform)
		test_dataloader = DataLoader (test_dataset, batch_size=args.batch_sz, shuffle=False)

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

		predictions = evaluate (args, config, tokenizer, av_enc_model, text_dec, audio_dec, video_dec, gen_head, test_dataloader, device)

		if args.last:
			out_file_path = config.output_path / f'last_predictions_{args.strategy}.json'
		elif args.pretrained:
			out_file_path = config.output_path / f'pretrained_predictions_{args.strategy}.json'
		else:
			out_file_path = config.output_path / f'best_predictions_{args.strategy}.json'

		with open (out_file_path, 'w') as file_io:
			json.dump (predictions, file_io)
			print (f'Predictions saved to {out_file_path}')	

		print ('Done !')

