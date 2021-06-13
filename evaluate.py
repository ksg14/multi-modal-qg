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


def evaluate (args, config, tokenizer, av_enc_model, text_enc_model, dec_model, dataloader, device):
	# val_bleu = 0.0
	# val_bleu_1 = 0.0
	# val_bleu_2 = 0.0
	# val_bleu_3 = 0.0
	# val_bleu_4 = 0.0
	n_len = len (dataloader)
	predictions = []

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

				enc_hidden_state, enc_attn = text_enc_model (context)
				
				if args.logs:
					print (f'audio emb - {audio_emb.shape}')
					print (f'video emb - {video_emb.shape}')
					print (f'enc hidden - {enc_hidden_state.shape}')

				enc_out = torch.cat ([enc_hidden_state, audio_emb.unsqueeze (0), video_emb.unsqueeze (0)], dim=1)

				if args.logs:
					print (f'enc out - {enc_out.shape}')

				if args.strategy == 'greedy':
					pred_question_ids = dec_model.decoder.generate (encoder_hidden_states=enc_out, max_length=args.max_len)
				elif args.strategy == 'beam':
					pred_question_ids = dec_model.decoder.generate (encoder_hidden_states=enc_out, max_length=args.max_len, num_beams=args.beams, early_stopping=True)
				
				print (f'pred ids - {type (pred_question_ids)}')

				pred_question_str = tokenizer.decode(pred_question_ids [0], skip_special_tokens=True)

				predictions.append ({
					'question_id' : question_id,
					'gt_question' : question_str, 
					'pred_question' : pred_question_str
				})


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

	try:
		config = Config (args.config_path)
	except Exception as e:
		print (f' Config load error {str (e)}')
		config = None

	if config:
		device = torch.device(args.device)
		print(f'Device - {device}')

		tokenizer = ProphetNetTokenizer.from_pretrained (config.pretrained_tokenizer_path)
		video_transform = T.Compose ([ToFloatTensor (), Resize (112)])
		
		test_dataset = VQGDataset (config.test_file, config.vocab_file, config.index_to_word_file, config.salient_frames_path, config.salient_audio_path, text_transform= None, prophetnet_transform=tokenizer, video_transform=video_transform)
		test_dataloader = DataLoader (test_dataset, batch_size=args.batch_sz, shuffle=False)

	
		av_enc_model = AudioVideoEncoder (config.av_in_channels, config.av_kernel_sz, config.av_stride, config.video_hidden_dim, config.flatten_dim, config.audio_emb, config.prophetnet_hidden_sz, device)
		
		if args.last:
			av_enc_model.load_state_dict(torch.load(config.output_path / 'last_av_model.pth', map_location=device))
		else:
			av_enc_model.load_state_dict(torch.load(config.av_model_path, map_location=device))
		
		av_enc_model.eval ()

		if args.last:
			enc_path = config.last_text_enc_model_path
		else:
			enc_path = config.text_enc_model_path

		text_enc_model = ProphetNetTextEncoder (enc_path)
		text_enc_model.eval()

		if args.last:
			dec_path = config.last_dec_model_path
		else:
			dec_path = config.dec_model_path

		dec_model = ProphetNetDecoder (config.pretrained_decoder_path)
		dec_model.eval ()

		av_enc_model.to (device)
		text_enc_model.to (device)
		dec_model.to (device)

		predictions = evaluate (args, config, tokenizer, av_enc_model, text_enc_model, dec_model, test_dataloader, device)

		if args.last:
			out_file_path = config.output_path / f'last_predictions_{args.strategy}.json'
		else:
			out_file_path = config.output_path / f'best_predictions_{args.strategy}.json'

		with open (out_file_path, 'w') as file_io:
			json.dump (predictions, file_io)
			print (f'Predictions saved to {out_file_path}')	

		print ('Done !')

