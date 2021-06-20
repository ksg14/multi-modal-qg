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

        if not os.path.exists(self.data_path):
            os.makedirs(self.data_path)

    # results
    output_path = Path (r'results/exp-enc_dec-1/')
    av_model_path = output_path / 'av_model.pth'
    text_enc_model_path = output_path / 'text_enc_model.pth'
    dec_model_path = output_path / 'dec_model.pth'
    stats_json_path = output_path / 'stats.json'
    stats_pkl_path = output_path / 'stats.pkl'
    learned_weight_path = output_path / 'learned_weight.pt'

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
    data_path = Path (r'data')
    vocab_file = data_path / 'vocab.json'
    index_to_word_file = data_path / 'index_to_word.json'
    weights_matrix_file = data_path / 'weight_matrix.npy'
    char_vocab_file = data_path / 'char_vocab.json'
    index_to_char_file = data_path / 'index_to_char.json'
    char_weights_matrix_file = data_path / 'char_weight_matrix.npy'
    preprocessed_text_file = data_path / 'preprocesses_text.json'

    # Squad
    squad_path = Path (r'squad')
    squad_train_file = squad_path / 'train-v2.0.json'
    squad_val_file = squad_path / 'dev-v2.0.json'
    squad_prep_train_file = squad_path / 'prep_train.json'
    squad_prep_val_file = squad_path / 'prep_val.json'

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
    glove_char_file = glove_path / f'glove.6B.{glove_emb_dim}d_char.txt'
    glove_chars_file = glove_path / f'6B.{glove_emb_dim}_chars.pkl'
    glove_char2idx_file = glove_path / f'6B.{glove_emb_dim}_char2idx.pkl'
    glove_char_matrix_file = glove_path / f'6B.{glove_emb_dim}_char_matrix.npy'

    # hyper-params
    optim='adam' # sgd, adam
    audio_emb = 128
    av_emb = 128 + 400
    vid_mean = [0.43216, 0.394666, 0.37645]
    vid_std = [0.22803, 0.22145, 0.216989]
    question_max_length = 22
    context_max_length = 200
    char_question_max_length = 185
    char_context_max_length = 3063
    av_max_length = 47
    # Video encoder
    av_in_channels = 3
    av_kernel_sz = 3
    av_stride = 1
    video_hidden_dim = 512
    video_emb_dim = 512
    # flatten_dim = 1000
    # text encoder
    text_lstm_hidden_dim = 512
    text_emb_dim = text_lstm_hidden_dim * 2
    text_lstm_layers = 2
    text_lstm_dropout = 0.1
    text_non_trainable = True
    # decoder
    dec_lstm_hidden_dim = 512
    dec_lstm_layers = 2
    dec_lstm_dropout = 0.1
    
    # checkpoints
    pretrained_models = Path (r'pretrained_models')
    pretrained_av_model = pretrained_models / 'av_model.pth'
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




    