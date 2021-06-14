from math import inf
import torch
from torch.nn import Module, LSTM, Linear, Dropout, GRU, Embedding
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_, orthogonal_, normal_

from transformers import ProphetNetForCausalLM, ProphetNetForConditionalGeneration
from transformers.file_utils import ModelOutput

class ProphetNetCG (Module):
    def __init__(self, dec_path, out_attentions=False):
        super().__init__()
        self.out_attentions = out_attentions

        self.model = ProphetNetForConditionalGeneration.from_pretrained (dec_path)
    
    def forward (self, context, src, tgt):
        outputs = self.model (input_ids=context.view (1, -1), decoder_input_ids=src.view (1, -1), labels=tgt.view (1, -1), output_hidden_states=True)
        
        return outputs.logits, outputs.decoder_hidden_states [-1]
    
    def generate (self, context, strategy, beams, max_len):
        if strategy == 'greedy':
            ids = self.model.generate (input_ids=context.view (1, -1), max_length=max_len) # return_dict_in_generate=True
        elif strategy == 'beam':
            ids = self.model.generate (input_ids=context.view (1, -1), max_length=max_len, num_beams=beams, early_stopping=True)

        return ids 

    def save_model (self, save_path):
        print (f'Saving model to {save_path}')
        self.model.save_pretrained(save_path)

class Decoder (Module):
    def __init__ (self, num_layers, dropout, hidden_dim, n_vocab, word_emb_dim, av_emb_dim, emb_layer):
        super().__init__()

        self.num_layers = num_layers
        self.dropout = dropout 
        self.hidden_dim = hidden_dim
        self.n_vocab = n_vocab
        self.word_emb_dim = word_emb_dim
        self.av_emb_dim = av_emb_dim
        self.word_embeddings = emb_layer

        self.lstm = LSTM(self.word_emb_dim + self.av_emb_dim, self.hidden_dim, self.num_layers, dropout=self.dropout)

        self.out_layer = Linear(self.hidden_dim, self.n_vocab)

        self.initialise_weights ()
        
    def forward (self, text, av_enc_out, hidden):
        word_emb = self.word_embeddings(text).view (text.shape [1], -1)

        word_av_emb = torch.cat ((word_emb, av_enc_out.repeat (text.shape [1], 1)), dim=1)

        lstm_out, hidden = self.lstm(word_av_emb.view(word_av_emb.shape [0], 1, -1), hidden)

        logits = self.out_layer (lstm_out)
        return logits, hidden
    
    def init_state(self, batch_sz):
        return (torch.zeros(self.num_layers, batch_sz, self.hidden_dim),
                torch.zeros(self.num_layers, batch_sz, self.hidden_dim))
    
    def initialise_weights (self):
        for param in self.lstm.parameters():
            if len(param.shape) >= 2:
                orthogonal_(param.data)
            else:
                normal_(param.data)
        
        xavier_uniform_ (self.out_layer.weight)
        normal_ (self.out_layer.bias)

class AttnDecoder (Module):
    def __init__(self, num_layers, dropout_p, hidden_dim, n_vocab, word_emb_dim, video_emb_dim, audio_emb_dim, emb_layer, text_max_length, av_max_length, device):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.n_vocab = n_vocab
        self.dropout_p = dropout_p
        self.text_max_length = text_max_length
        self.av_max_length = av_max_length
        self.video_emb_dim = video_emb_dim
        self.audio_emb_dim = audio_emb_dim
        self.word_emb_dim = word_emb_dim
        self.emb_layer = emb_layer
        self.device = device

        self.text_attn = Linear (self.word_emb_dim + self.hidden_dim, self.text_max_length)
        self.vid_attn = Linear (self.word_emb_dim + self.hidden_dim, self.av_max_length)
        self.audio_attn = Linear (self.word_emb_dim + self.hidden_dim, self.av_max_length)
        # self.attn_combine = Linear (self.word_emb_dim + self.hidden_dim + self.av_emb_dim, self.hidden_dim)
        self.dropout = Dropout (self.dropout_p)
        self.lstm = LSTM (self.word_emb_dim + self.hidden_dim + self.audio_emb_dim + self.video_emb_dim, self.hidden_dim, self.num_layers, dropout=self.dropout_p)
        self.out_layer = Linear (self.hidden_dim, self.n_vocab)

        self.initialise_weights ()

    def forward(self, word, enc_frames, enc_seq_len, audio_emb, video_emb, hidden, encoder_outputs):
        embedded = self.emb_layer (word).view(1, 1, -1)

        # Text attention
        text_attn_pre_soft = self.text_attn(torch.cat((embedded[0], hidden[0] [-1]), 1))
        text_attn_pre_soft [enc_seq_len:] = float ('-inf')
        text_attn_weights = F.softmax(text_attn_pre_soft, dim=1)
        text_attn_applied = torch.bmm(text_attn_weights.unsqueeze(0), encoder_outputs.unsqueeze(0))

        # Video attention
        vid_attn_pre_soft = self.vid_attn(torch.cat((embedded[0], hidden[0] [-1]), 1))
        vid_attn_pre_soft [enc_frames:] = float ('-inf')
        vid_attn_weights = F.softmax(vid_attn_pre_soft, dim=1)
        vid_attn_applied = torch.bmm(vid_attn_weights.unsqueeze(0), video_emb.unsqueeze(0))

        print (f'audio_emb - {audio_emb.shape}')

        # Audio attention
        audio_attn_pre_soft = self.audio_attn(torch.cat((embedded[0], hidden[0] [-1]), 1))
        audio_attn_pre_soft [enc_frames:] = float ('-inf')
        audio_attn_weights = F.softmax(audio_attn_pre_soft, dim=1)
        audio_attn_applied = torch.bmm(audio_attn_weights.unsqueeze(0), audio_emb.unsqueeze(0))

        print (f'audio_attn_applied {audio_attn_applied.shape}')

        output = torch.cat((embedded[0], text_attn_applied[0], audio_attn_applied [0], vid_attn_applied [0]), 1)
        # output = self.attn_combine(output).unsqueeze(0)
        output = output.unsqueeze (0)

        # output = F.relu(output)
        output, hidden = self.lstm (output, hidden)

        output = self.out_layer(output[0])
        return output, hidden, text_attn_weights, audio_attn_weights, vid_attn_weights
    
    def initialise_weights (self):
        for param in self.lstm.parameters():
            if len(param.shape) >= 2:
                orthogonal_(param.data)
            else:
                normal_(param.data)
        
        xavier_uniform_ (self.out_layer.weight)
        normal_ (self.out_layer.bias)
        xavier_uniform_ (self.text_attn.weight)
        normal_ (self.text_attn.bias)
        xavier_uniform_ (self.audio_attn.weight)
        normal_ (self.audio_attn.bias)
        xavier_uniform_ (self.vid_attn.weight)
        normal_ (self.vid_attn.bias)
        # xavier_uniform_ (self.attn_combine.weight)
        # normal_ (self.attn_combine.bias)

class AudioDecoder (Module):
    def __init__(self, num_layers, dropout_p, hidden_dim, n_vocab, word_emb_dim, audio_emb_dim, emb_layer, av_max_length, device):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.n_vocab = n_vocab
        self.dropout_p = dropout_p
        self.av_max_length = av_max_length
        self.audio_emb_dim = audio_emb_dim
        self.word_emb_dim = word_emb_dim
        self.emb_layer = emb_layer
        self.device = device

        self.audio_attn = Linear (self.word_emb_dim + self.hidden_dim, self.av_max_length)
        # self.attn_combine = Linear (self.word_emb_dim + self.hidden_dim + self.av_emb_dim, self.hidden_dim)
        self.dropout = Dropout (self.dropout_p)
        self.lstm = LSTM (self.word_emb_dim + self.audio_emb_dim, self.hidden_dim, self.num_layers, dropout=self.dropout_p)
        self.out_layer = Linear (self.hidden_dim, self.n_vocab)

        self.initialise_weights ()

    def forward(self, word, enc_frames, audio_emb, hidden):
        embedded = self.emb_layer (word).view(1, 1, -1)

        # Audio attention
        audio_attn_pre_soft = self.audio_attn(torch.cat((embedded[0], hidden[0] [-1]), 1))
        audio_attn_pre_soft [enc_frames:] = float ('-inf')
        audio_attn_weights = F.softmax(audio_attn_pre_soft, dim=1)
        audio_attn_applied = torch.bmm(audio_attn_weights.unsqueeze(0), audio_emb.unsqueeze(0))

        # print (f'audio_attn_applied {audio_attn_applied.shape}')

        output = torch.cat((embedded[0], audio_attn_applied [0]), 1)
        # output = self.attn_combine(output).unsqueeze(0)
        output = output.unsqueeze (0)

        # output = F.relu(output)
        output, hidden = self.lstm (output, hidden)

        output = self.out_layer(output[0])
        return output, hidden, audio_attn_weights

    def init_state(self, batch_sz):
        return (torch.zeros(self.num_layers, batch_sz, self.hidden_dim, device=self.device),
                torch.zeros(self.num_layers, batch_sz, self.hidden_dim, device=self.device))
    
    def initialise_weights (self):
        for param in self.lstm.parameters():
            if len(param.shape) >= 2:
                orthogonal_(param.data)
            else:
                normal_(param.data)
        
        xavier_uniform_ (self.out_layer.weight)
        normal_ (self.out_layer.bias)
        xavier_uniform_ (self.audio_attn.weight)
        normal_ (self.audio_attn.bias)
        # xavier_uniform_ (self.attn_combine.weight)
        # normal_ (self.attn_combine.bias)

class VideoDecoder (Module):
    def __init__(self, num_layers, dropout_p, hidden_dim, n_vocab, word_emb_dim, video_emb_dim, emb_layer, av_max_length, device):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.n_vocab = n_vocab
        self.dropout_p = dropout_p
        self.av_max_length = av_max_length
        self.word_emb_dim = word_emb_dim
        self.video_emb_dim = video_emb_dim
        self.emb_layer = emb_layer
        self.device = device

        self.vid_attn = Linear (self.word_emb_dim + self.hidden_dim, self.av_max_length)
        # self.attn_combine = Linear (self.word_emb_dim + self.hidden_dim + self.av_emb_dim, self.hidden_dim)
        self.dropout = Dropout (self.dropout_p)
        self.lstm = LSTM (self.word_emb_dim + self.video_emb_dim, self.hidden_dim, self.num_layers, dropout=self.dropout_p)
        self.out_layer = Linear (self.hidden_dim, self.n_vocab)

        self.initialise_weights ()

    def forward(self, word, enc_frames, video_emb, hidden):
        embedded = self.emb_layer (word).view(1, 1, -1)

        # Video attention
        vid_attn_pre_soft = self.vid_attn(torch.cat((embedded[0], hidden[0] [-1]), 1))
        vid_attn_pre_soft [enc_frames:] = float ('-inf')
        vid_attn_weights = F.softmax(vid_attn_pre_soft, dim=1)
        vid_attn_applied = torch.bmm(vid_attn_weights.unsqueeze(0), video_emb.unsqueeze(0))

        output = torch.cat((embedded[0], vid_attn_applied [0]), 1)
        # output = self.attn_combine(output).unsqueeze(0)
        output = output.unsqueeze (0)

        # output = F.relu(output)
        output, hidden = self.lstm (output, hidden)

        output = self.out_layer(output[0])
        return output, hidden, vid_attn_weights
    
    def init_state(self, batch_sz):
        return (torch.zeros(self.num_layers, batch_sz, self.hidden_dim, device=self.device),
                torch.zeros(self.num_layers, batch_sz, self.hidden_dim, device=self.device))
    
    def initialise_weights (self):
        for param in self.lstm.parameters():
            if len(param.shape) >= 2:
                orthogonal_(param.data)
            else:
                normal_(param.data)
        
        xavier_uniform_ (self.out_layer.weight)
        normal_ (self.out_layer.bias)
        xavier_uniform_ (self.vid_attn.weight)
        normal_ (self.vid_attn.bias)
        # xavier_uniform_ (self.attn_combine.weight)
        # normal_ (self.attn_combine.bias)

class GenerationHead (Module):
    def __init__(self, enc_emb_dim, n_vocab, device):
        super(AttnDecoder, self).__init__()
        self.enc_emb_dim = enc_emb_dim
        self.n_vocab = n_vocab
        self.device = device

        self.out_layer = Linear (self.enc_emb_dim, self.n_vocab)

        self.initialise_weights ()

    def forward(self, audio_out, video_out, text_out):
        output = self.out_layer(torch.cat ([audio_out, video_out, text_out], dim=1))
        return output
    
    def initialise_weights (self):
        xavier_uniform_ (self.out_layer.weight)
        normal_ (self.out_layer.bias)
