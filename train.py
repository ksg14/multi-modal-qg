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

def validate (av_enc_model, text_enc_model, dec_model, dataloader, context_max_len, pred_max_len, device):
    # val_loss = 0.0
    val_bleu = 0.0
    val_bleu_1 = 0.0
    val_bleu_2 = 0.0
    val_bleu_3 = 0.0
    val_bleu_4 = 0.0
    n_len = len (dataloader)
    
    av_enc_model.eval () 
    text_enc_model.eval ()
    dec_model.eval ()

    with torch.no_grad ():
        with tqdm(dataloader) as tepoch:
            for frames, audio_file, context_tensor, question_id, question, target, context_len, target_len in tepoch:
                frames, audio_file, context_tensor, question_id, question, target, context_len, target_len = frames.to (device), audio_file, context_tensor.to (device), question_id, question, target.to (device), context_len.to (device), target_len.to (device)
                
                tepoch.set_description (f'Validating ...')

                av_enc_out = av_enc_model (audio_file [0], frames)

                text_enc_hidden = text_enc_model.init_state (1)
                all_enc_outputs = torch.zeros(context_max_len, 128).to (device)

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
                tepoch.set_postfix (val_bleu=val_bleu)
    
    print (f'Val_bleu - {round (val_bleu, 3)}, Val_bleu_1 - {round (val_bleu_1, 3)}')
    return val_bleu / n_len, val_bleu_1 / n_len, val_bleu_2 / n_len, val_bleu_3 / n_len, val_bleu_4 / n_len 

def train (av_enc_model, text_enc_model, dec_model, train_dataloader, val_dataloader, text_enc_optimizer, dec_optimizer, criterion, n_epochs, context_max_len, pred_max_len, device):
    epoch_stats = { 'train' : {'loss' : []}, 'val' : {'loss' : [], 'bleu' : [], 'bleu_1' : [], 'bleu_2' : [], 'bleu_3' : [], 'bleu_4' : []} }
    n_len = len (train_dataloader)
    best_bleu = -1

    for epoch in range (n_epochs):
        epoch_stats ['train']['loss'].append (0.0)
        # av_enc_model.train ()
        text_enc_model.train ()
        dec_model.train ()

        with tqdm(train_dataloader) as tepoch:
            for frames, audio_file, context_tensor, question_id, question, target, context_len, target_len in tepoch:
                frames, audio_file, context_tensor, question_id, question, target, context_len, target_len = frames.to (device), audio_file, context_tensor.to (device), question_id, question, target.to (device), context_len.to (device), target_len.to (device)
                
                tepoch.set_description (f'Epoch {epoch}')

                # av_enc_optimizer.zero_grad()
                text_enc_optimizer.zero_grad ()
                dec_optimizer.zero_grad()

                av_enc_out = av_enc_model (audio_file [0], frames)

                text_enc_hidden = text_enc_model.init_state (1)
                all_enc_outputs = torch.zeros(context_max_len, text_enc_model.hidden_dim).to (device)

                loss = 0

                # print (f'av_enc_out shape - {av_enc_out.shape}')
                # print (f'context_len - {context_len}')
                # print (f'target_len - {target_len}')
                # print (f'av enc out - {av_enc_out.shape}')
                # print (f'context shape - {context_tensor.shape}')
                # print (f'target shape - {target.shape}')
                # print (f'context [0] - {context_tensor [0] [0]}')

                for ei in range (context_len):
                    enc_output, text_enc_hidden = text_enc_model(context_tensor [0][ei], text_enc_hidden)
                    all_enc_outputs [ei] = enc_output [0, 0]
                
                # print (all_enc_outputs.shape)
                # print (all_enc_outputs.max ().item ())
                # print (all_enc_outputs.min ().item ())

                dec_input = torch.tensor([[train_dataloader.dataset.vocab ['<start>']]]).to (device)
                dec_hidden = text_enc_hidden

                # y_pred, dec_hidden = dec_model (question, av_enc_out, dec_hidden)

                for di in range (target_len):
                    dec_output, dec_hidden, dec_attention = dec_model (dec_input, context_len, av_enc_out, dec_hidden, all_enc_outputs)
                    loss += criterion (dec_output, target [0][di].view (-1))
                    dec_input = target [0] [di]  # Teacher forcing

                loss.backward()

                # av_enc_optimizer.step()
                text_enc_optimizer.step ()
                dec_optimizer.step()

                with torch.no_grad():
                    epoch_stats ['train']['loss'] [-1] += ((loss.item () / target_len) / n_len).item ()
                
                tepoch.set_postfix (train_loss=epoch_stats ['train']['loss'] [-1])
                # break
        # break
        val_bleu, val_bleu_1, val_bleu_2, val_bleu_3, val_bleu_4 = validate (av_enc_model, text_enc_model, dec_model, val_dataloader, context_max_len, pred_max_len, device)
        # epoch_stats ['val']['loss'].append (val_loss)
        epoch_stats ['val']['bleu'].append (val_bleu)
        epoch_stats ['val']['bleu_1'].append (val_bleu_1)
        epoch_stats ['val']['bleu_2'].append (val_bleu_2)
        epoch_stats ['val']['bleu_3'].append (val_bleu_3)
        epoch_stats ['val']['bleu_4'].append (val_bleu_4)

        # Save best model
        if val_bleu > best_bleu:
            best_bleu = val_bleu

            print ('Saving new best model !')
            # save_model (av_enc_model, config.av_model_path)
            save_model (text_enc_model, config.text_enc_model_path)
            save_model (dec_model, config.dec_model_path)
            save_weights (dec_model.emb_layer, config.learned_weight_path)

        print({ 'epoch': epoch, 'train_loss': epoch_stats ['train']['loss'] [-1] })
        # break
    return epoch_stats

if __name__ == '__main__':
    config = Config ()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f'Device - {device}')

    weights_matrix = torch.from_numpy(np.load (config.weights_matrix_file))
    weights_matrix = weights_matrix.long ().to (device)
    
    video_transform = T.Compose ([ToFloatTensor (), Resize (112), Normalize (config.vid_mean, config.vid_std)])

    train_dataset = VQGDataset (config.train_file, config.vocab_file, config.index_to_word_file, config.salient_frames_path, config.salient_audio_path, text_transform= prepare_sequence, video_transform=video_transform)
    val_dataset = VQGDataset (config.val_file, config.vocab_file, config.index_to_word_file, config.salient_frames_path, config.salient_audio_path, text_transform= prepare_sequence, video_transform=video_transform)
    train_dataloader = DataLoader (train_dataset, batch_size=1, shuffle=False)
    val_dataloader = DataLoader (val_dataset, batch_size=1, shuffle=False)

    emb_layer, n_vocab, emb_dim = create_emb_layer (weights_matrix, False)    

    av_enc_model = AudioVideoEncoder ()
    av_enc_model.eval ()

    text_enc_model = TextEncoder (num_layers=config.text_lstm_layers, \
                        dropout=config.text_lstm_dropout, \
                        hidden_dim=config.text_lstm_hidden_dim, \
                        emb_dim=emb_dim, \
                        emb_layer=emb_layer, \
                        device=device)
    
    dec_model = AttnDecoder (num_layers=config.dec_lstm_layers, \
                        dropout_p=config.dec_lstm_dropout, \
                        hidden_dim=config.dec_lstm_hidden_dim, \
                        n_vocab=n_vocab, \
                        word_emb_dim=emb_dim, \
                        av_emb_dim=config.av_emb, \
                        emb_layer=emb_layer, \
                        max_length=config.context_max_lenth, \
                        device=device)

    criterion = CrossEntropyLoss()
    # av_enc_optimizer = Adam(av_enc_model.parameters(), lr=0.001)
    text_enc_optimizer = Adam(text_enc_model.parameters(), lr=config.lr)
    dec_optimizer = Adam(dec_model.parameters(), lr=config.lr)

    av_enc_model.to (device)
    text_enc_model.to (device)
    dec_model.to (device)

    epoch_stats = train (av_enc_model, text_enc_model, dec_model, train_dataloader, val_dataloader, text_enc_optimizer, dec_optimizer, criterion, config.epochs, device=device, context_max_len=config.context_max_lenth, pred_max_len=config.question_max_length)

    # validate (av_enc_model, text_enc_model, dec_model, val_dataloader, context_max_lenth, question_max_length, device)

    try:
        with open (config.stats_json_path, 'w') as f:
            json.dump (epoch_stats, f)
            print (f'Stats saved to {config.stats_json_path}')
    except Exception:
        pickle.dump(epoch_stats, open(config.stats_pkl_path, 'wb'))
        print (f'Stats saved to {config.stats_pkl_path}')
    
    try:
        config.save_config ()
    except Exception:
        print (f'Unable to save config {str (Exception)}')
    
    
    print ('Done !')

    # print (f'mem av - {get_mem_usage (av_enc_model)} MB')
    # print (f'mem text enc - {get_mem_usage (text_enc_model)} MB')
    # print (f'mem dec - {get_mem_usage (dec_model)} MB')

