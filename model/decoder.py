from math import inf
import torch
from torch.nn import Module, LSTM, Linear, Dropout, GRU, Embedding
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_, orthogonal_, normal_

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
    def __init__(self, num_layers, dropout_p, hidden_dim, n_vocab, word_emb_dim, av_emb_dim, emb_layer, max_length, device):
        super(AttnDecoder, self).__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.n_vocab = n_vocab
        self.dropout_p = dropout_p
        self.max_length = max_length
        self.av_emb_dim = av_emb_dim
        self.word_emb_dim = word_emb_dim
        self.emb_layer = emb_layer
        self.device = device

        self.attn = Linear (self.word_emb_dim + self.hidden_dim, self.max_length)
        self.attn_combine = Linear (self.word_emb_dim + self.hidden_dim + self.av_emb_dim, self.hidden_dim)
        self.dropout = Dropout (self.dropout_p)
        self.lstm = LSTM (self.hidden_dim, self.hidden_dim)
        self.out_layer = Linear (self.hidden_dim, self.n_vocab)

        self.initialise_weights ()

    def forward(self, word, enc_seq_len, av_emb, hidden, encoder_outputs):
        embedded = self.emb_layer (word).view(1, 1, -1)

        attn_pre_soft = self.attn(torch.cat((embedded[0], hidden[0] [0]), 1))
        attn_pre_soft [enc_seq_len:] = float ('-inf')

        attn_weights = F.softmax(attn_pre_soft, dim=1)

        attn_applied = torch.bmm(attn_weights.unsqueeze(0), encoder_outputs.unsqueeze(0))

        output = torch.cat((embedded[0], attn_applied[0], av_emb), 1)
        output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output)
        output, hidden = self.lstm (output, hidden)

        output = self.out_layer(output[0])
        return output, hidden, attn_weights
    
    def initialise_weights (self):
        for param in self.lstm.parameters():
            if len(param.shape) >= 2:
                orthogonal_(param.data)
            else:
                normal_(param.data)
        
        xavier_uniform_ (self.out_layer.weight)
        normal_ (self.out_layer.bias)
        xavier_uniform_ (self.attn.weight)
        normal_ (self.attn.bias)
        xavier_uniform_ (self.attn_combine.weight)
        normal_ (self.attn_combine.bias)