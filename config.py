from pathlib import Path, PurePath
import json
import os

class Config():
    def __init__(self, config_path=None):
        if config_path:
            with open (config_path, 'r') as f:
                config_data = json.load (f)
                self.load_config (**config_data)

        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)
        
        if not os.path.exists(self.text_model_path):
            os.makedirs(self.text_model_path)
        
        if not os.path.exists(self.last_text_model_path):
            os.makedirs(self.last_text_model_path)

        if not os.path.exists(self.data_path):
            os.makedirs(self.data_path)

    # results
    output_path = Path (r'results/exp-vqg_multi-1/')
    av_model_path = output_path / 'av_model.pth'
    text_model_path = output_path / 'text_model'
    audio_model_path = output_path / 'audio_model.pth'
    video_model_path = output_path / 'video_model.pth'
    gen_head_model_path = output_path / 'gen_head_model.pth'
    last_text_model_path = output_path / 'last_text_model'
    stats_json_path = output_path / 'stats.json'
    stats_pkl_path = output_path / 'stats.pkl'
    learned_weight_path = output_path / 'learned_weight.pt'

    # pretrained models
    pretrained_path = Path (f'pretrained_models')
    pretrained_tokenizer_path = pretrained_path / 'tokenizer'
    pretrained_encoder_path = pretrained_path / 'encoder'
    pretrained_decoder_path = pretrained_path / 'decoder'
    pretrained_cg_path = pretrained_path / 'prophetnet_cg'
    pretrained_cg_enc_path = pretrained_path / 'prophetnet_cg_enc'
    pretrained_cg_dec_path = pretrained_path / 'prophetnet_cg_dec'


    # dataset
    # dataset_path = Path (r'C:\Users\karanjit.singh.gill\Desktop\VQG\dataset')
    dataset_path = Path (r'dataset')
    subs_path = dataset_path / 'subs'
    video_path = dataset_path / 'vids'
    audio_path = dataset_path / 'audio'
    salient_text_path = dataset_path / 'salient_text'
    salient_frames_path = dataset_path / 'salient_frames'
    salient_audio_path = dataset_path / 'salient_audio_clip'
    salient_text_file = salient_text_path / 'salient_text_list.json'
    questions_file = dataset_path / 'labelled_questions.json'
    videos_file = dataset_path / 'videos.json'

    # data
    data_path = Path ('data')
    vocab_file = data_path / 'vocab.json'
    index_to_word_file = data_path / 'index_to_word.json'
    weights_matrix_file = data_path / 'weight_matrix.npy'
    preprocessed_text_file = data_path / 'preprocesses_text.json'

    # train/val/test
    train_file = data_path / 'train_questions.json'
    val_file = data_path / 'val_questions.json'
    test_file = data_path / 'test_questions.json'

    # glove
    glove_emb_dim = 300
    # glove_path = Path (r'C:\Users\karanjit.singh.gill\Desktop\VQG\saliency_transcript\glove.6B')
    glove_path = Path (r'glove.6B')
    glove_file = glove_path / f'glove.6B.{glove_emb_dim}d.txt'
    glove_words_file = glove_path / f'6B.{glove_emb_dim}_words.pkl'
    glove_idx_file = glove_path / f'6B.{glove_emb_dim}_idx.pkl'
    glove_matrix_file = glove_path / f'6B.{glove_emb_dim}_matrix.npy'

    # hyper-params
    optim='adam' # sgd, adam
    question_max_length = 21
    context_max_lenth = 283
    av_max_length = 101
    
    # Audio
    audio_emb = 128
    av_emb = 128 + 400
    audio_dec_hidden = 256
    audio_dec_layers = 1
    audio_dec_dropout = 0.2
    
    # Video
    vid_mean = [0.43216, 0.394666, 0.37645]
    vid_std = [0.22803, 0.22145, 0.216989]
    av_in_channels = 3
    av_kernel_sz = 3
    av_stride = 1
    video_hidden_dim = 512
    flatten_dim = 1000
    video_dec_hidden = 256
    video_dec_layers = 1
    video_dec_dropout = 0.2
    
    # prophetnet
    prophetnet_hidden_sz = 1024
    prophetnet_vocab = 30522

    # checkpoints
    best_epoch = None

    def save_config (self):
        attributes = [ key for key in Config.__dict__ if key [0] != '_' and not callable(Config.__dict__ [key])]
        save_data = {}

        for key in attributes:
            if isinstance(Config.__dict__ [key], PurePath):
                save_data [key] = str (Config.__dict__ [key])
            else:
                save_data [key] = Config.__dict__ [key]
   
        with open (self.output_path / 'config.json', 'w') as f:
            json.dump (save_data, f)
        return

    def load_config (self, **kwargs):
        class_attributes = [ key for key in Config.__dict__ if key [0] != '_' and not callable(Config.__dict__ [key])]

        for key, value in kwargs.items():
            if key in class_attributes:
                if isinstance (value, str) and key != 'optim':
                    setattr (Config, key, Path (value))
                else:
                    setattr (Config, key, value)
        # print (Config.__dict__)
        return

# if __name__ == '__main__':
#     test = Config ()
#     test.save_config ()




    