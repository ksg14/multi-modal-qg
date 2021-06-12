import os
import argparse

from transformers import ProphetNetTokenizer, ProphetNetEncoder, ProphetNetForConditionalGeneration

from config import Config

def save_encoder (config: Config) -> int:
	try:
		tokenizer = ProphetNetTokenizer.from_pretrained('microsoft/prophetnet-large-uncased-squad-qg')
		model = ProphetNetEncoder.from_pretrained('microsoft/prophetnet-large-uncased-squad-qg')
	
		tokenizer.save_pretrained(config.pretrained_tokenizer_path)
		model.save_pretrained(config.pretrained_encoder_path)
	except Exception as e:
		print (f'Error - {str (e)}')
		return 1
	return 0

def save_cg_decoder (config: Config) -> int:
	try:
		model = ProphetNetForConditionalGeneration.from_pretrained("microsoft/prophetnet-large-uncased-squad-qg", is_decoder = True, add_cross_attention=True)

		model.save_pretrained(config.pretrained_decoder_path)
	except Exception as e:
		print (f'Error - {str (e)}')
		return 1
	return 0

if __name__ == '__main__' :
	parser = argparse.ArgumentParser(description='Get caption len stats')
	parser.add_argument('-e',
                       '--encoder',
                       action='store_true',
                       help='get pretrained encoder')
	parser.add_argument('-d',
                       '--decoder',
                       action='store_true',
                       help='get pretrained decoder')
	
	args = parser.parse_args()

	config = Config ()
	
	if args.encoder:
		print (f'Saving encoder model')

		if not os.path.exists (config.pretrained_path):
			os.mkdir (config.pretrained_path)
		
		if not os.path.exists (config.pretrained_tokenizer_path):
			os.mkdir (config.pretrained_tokenizer_path)
		
		if not os.path.exists (config.pretrained_encoder_path):
			os.mkdir (config.pretrained_encoder_path)
	
		save_encoder (config)
	
	if args.decoder:
		print (f'Saving decoder model')

		if not os.path.exists (config.pretrained_path):
			os.mkdir (config.pretrained_path)
		
		if not os.path.exists (config.pretrained_decoder_path):
			os.mkdir (config.pretrained_decoder_path)
		
		# save_electra_decoder (config)
		save_cg_decoder (config)
	
	print ('Done!')