import os
import torch
import os

from model.encoder import AudioVideoEncoder
from train import save_model

from config import Config

def save_vggish (config):
	av_enc_model = AudioVideoEncoder (config.av_in_channels, config.av_kernel_sz, config.av_stride, config.video_hidden_dim, config.flatten_dim)

	save_model (av_enc_model, config.pretrained_av_model)
	print (f'Saved AV model to {config.pretrained_av_model}')

if __name__ == '__main__':
	config = Config ()

	if not os.path.exists (config.pretrained_models):
		os.mkdir (config.pretrained_models)

	save_vggish (config)
	
	print ('Done!')
