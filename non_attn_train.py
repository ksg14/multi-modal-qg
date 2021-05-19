from typing import Text
from model.decoder import Decoder
import torch
from torch.nn import Embedding, CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import transforms as T

import numpy as np

from model.encoder import AudioVideoEncoder, TextEncoder
from model.decoder import Decoder

from config import Config
from utils.dataset import VQGDataset
from utils.custom_transforms import prepare_sequence, get_word_from_idx,Resize, ToFloatTensor, Normalize, prepare_sequence

def create_emb_layer (weights_matrix, non_trainable):
    num_embeddings, embedding_dim = weights_matrix.size()
    emb_layer = Embedding(num_embeddings, embedding_dim)
    emb_layer.load_state_dict({'weight': weights_matrix})
    if non_trainable:
        emb_layer.weight.requires_grad = False
    return emb_layer, num_embeddings, embedding_dim

def save_model (model, model_path):
    try:
        torch.save(model, model_path)
        print (f'Model saved to {model_path}')
    except Exception:
        print (f'unable to save model {str (Exception)}')
    return

def repackage_hidden(h):
    """Wraps hidden states in new Tensors, to detach them from their history."""
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)

def validate (enc_model, dec_model, dataloader, max_len):
    val_loss = 0.0
    val_bleu_1 = 0.0
    
    enc_model.eval ()
    dec_model.eval ()

    with torch.no_grad ():
        for _, (frames, audio_file, context_tensor, question, target, context_len, target_len) in enumerate (dataloader):
            # print (f'frame - {frames.shape}')
            # print (f'audio - {audio_file}')
            # print (f'context - {context_tensor.shape}')
            # print (f'target - {target.shape}')
            # print (f'context len - {context_len}')
            # print (f'target len - {target_len}')

            enc_out = enc_model (audio_file [0], frames, context_tensor)

            state_h, state_c = dec_model.init_state(1)

            pred_words = ['<start>']

            for i in range(max_len):
                x = prepare_sequence(pred_words [-1], dataloader.dataset.vocab)
                print (x)
                # break
                y_pred, (state_h, state_c) = dec_model(pred_words [-1], enc_out, (state_h, state_c))

                last_word_logits = y_pred[0][-1]
                p = torch.nn.functional.softmax(last_word_logits, dim=0).detach().numpy()
                word_index = np.random.choice(len(last_word_logits), p=p)
                pred_words.append(dataloader.dataset.index_to_word[word_index])

                break
            print (pred_words)

            break
    return 0, 0

def train (av_enc_model, text_enc_model, dec_model, train_dataloader, val_dataloader, av_enc_optimizer, text_enc_optimizer, dec_optimizer, criterion, n_epochs, pred_max_len):
    epoch_stats = { 'train' : {'loss' : []}, 'val' : {'loss' : [], 'bleu-1' : []} }
    n_len = len (train_dataloader)

    for epoch in range (n_epochs):
        epoch_stats ['train']['loss'].append (0.0)
        av_enc_model.train ()
        text_enc_model.train ()
        dec_model.train ()

        # for _, (frames, audio_file, context_tensor, question, target, context_len, target_len) in enumerate (train_dataloader):
        with tqdm(dataloader) as tepoch:
            for x, y in tepoch:
            # print (f'frame - {frames.shape}')
            # print (f'audio - {audio_file}')
            # print (f'context - {context_tensor.shape}')
            # print (f'target - {target.shape}')
            # print (f'context len - {context_len}')
            # print (f'target len - {target_len}')

            av_enc_optimizer.zero_grad()
            text_enc_optimizer.zero_grad ()
            dec_optimizer.zero_grad()

            # with torch.autograd.set_detect_anomaly(True):
            av_enc_out = av_enc_model (audio_file [0], frames)

            text_enc_hidden = text_enc_model.init_state (1)
            text_enc_out, text_enc_hidden = text_enc_model (context_tensor, text_enc_hidden)
            
            # print (f'av enc out - {av_enc_out.shape}')
            # print (f'text hidden final - {text_enc_hidden [0].shape}')

            dec_hidden = text_enc_hidden
            # print (f'dec hidden - {dec_hidden [0].shape}')

            # print (f'question - {question.shape}')

            y_pred, dec_hidden = dec_model (question, av_enc_out, dec_hidden)
            # print (f'ypred - {y_pred.shape}')
            # print (f'target - {target.shape}')
            # print (f'final token shape - {target [0][-1].view (-1).shape}')

            loss = criterion(y_pred, target [0][-1].view (-1))

            loss.backward()

            av_enc_optimizer.step()
            text_enc_optimizer.step ()
            dec_optimizer.step()

            with torch.no_grad():          
                epoch_stats ['train']['loss'] [-1] += loss.item () / n_len

            # target = target.squeeze ()
            # print (target.shape)

            # for i in range (question.shape [1]):
            #     av_enc_optimizer.zero_grad()
            #     text_enc_optimizer.zero_grad ()
            #     dec_optimizer.zero_grad()

            #     dec_hidden = repackage_hidden (dec_hidden)
            #     y_pred, dec_hidden = dec_model (question [0][i], enc_out, dec_hidden)
            #     print (y_pred.shape)
            #     print (target [i].view (-1).shape)

            #     loss = criterion(y_pred, target [i].view (-1))

            #     loss.backward()

            #     av_enc_optimizer.step()
            #     text_enc_optimizer.step ()
            #     dec_optimizer.step()

            #     with torch.no_grad():                    
            #         epoch_stats ['train']['loss'] [-1] += loss.item () / n_len
        # break

        # val_loss, val_bleu = validate (enc_model, dec_model, criterion, val_dataloader, pred_max_len)

        print({ 'epoch': epoch, 'loss': epoch_stats ['train']['loss'] [-1] })
        break
    
    return

if __name__ == '__main__':
    config = Config ()

    av_emb = 128 + 400 # + 128
    
    weights_matrix = torch.from_numpy(np.load (config.weights_matrix_file))
    weights_matrix = weights_matrix.long ()
    
    mean = [0.43216, 0.394666, 0.37645]
    std = [0.22803, 0.22145, 0.216989]
    video_transform = T.Compose ([ToFloatTensor (), Resize (112), Normalize (mean, std)])

    train_dataset = VQGDataset (config.train_file, config.vocab_file, config.index_to_word_file, config.salient_frames_path, config.salient_audio_path, text_transform= prepare_sequence, video_transform=video_transform)
    val_dataset = VQGDataset (config.val_file, config.vocab_file, config.index_to_word_file, config.salient_frames_path, config.salient_audio_path, text_transform= prepare_sequence, video_transform=video_transform)
    train_dataloader = DataLoader (train_dataset, batch_size=1, shuffle=False)
    val_dataloader = DataLoader (val_dataset, batch_size=1, shuffle=False)

    emb_layer, n_vocab, emb_dim = create_emb_layer (weights_matrix, False)    

    av_enc_model = AudioVideoEncoder ()

    text_enc_model = TextEncoder (num_layers=config.text_lstm_layers, \
                        dropout=config.text_lstm_dropout, \
                        hidden_dim=config.text_lstm_hidden_dim, \
                        emb_dim=emb_dim, \
                        emb_layer=emb_layer)
    
    dec_model = Decoder (num_layers=config.dec_lstm_layers, \
                        dropout=config.dec_lstm_dropout, \
                        hidden_dim=config.dec_lstm_hidden_dim, \
                        n_vocab=n_vocab, \
                        word_emb_dim=emb_dim, \
                        av_emb_dim=av_emb, \
                        emb_layer=emb_layer)

    criterion = CrossEntropyLoss()
    av_enc_optimizer = Adam(av_enc_model.parameters(), lr=0.001)
    text_enc_optimizer = Adam(text_enc_model.parameters(), lr=0.001)
    dec_optimizer = Adam(dec_model.parameters(), lr=0.001)

    train (av_enc_model, text_enc_model, dec_model, train_dataloader, val_dataloader, av_enc_optimizer, text_enc_optimizer, dec_optimizer, criterion, config.epochs, pred_max_len=15)
