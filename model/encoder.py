import torch
from torch._C import device
from torch.nn import Module, LSTM, Linear, AdaptiveAvgPool1d
# import torch.nn.functional as F
from torch.nn.init import xavier_uniform_, orthogonal_, normal_
from torch.nn.modules.conv import Conv1d

import torchvision.models as models

class AudioEncoder (Module):
    def __init__ (self):
        super().__init__()
        
        self.vggish = torch.hub.load('harritaylor/torchvggish', 'vggish', postprocess=False)
        self.adapt_avg_pool = AdaptiveAvgPool1d(1)
        # self.fc1 = Linear (128, audio_emb) 

    def forward (self, audio_file):
        out = self.vggish.forward (audio_file)
        embeddings = self.adapt_avg_pool (out.view (1, out.shape [1], -1))
        return embeddings

class VideoEncoder (Module):
    def __init__ (self, download_pretrained=False):
        super().__init__()
        self.resnet3d = models.video.r2plus1d_18 (pretrained=download_pretrained, progress=True)

    def forward (self, video_frames):
        embeddings = self.resnet3d (video_frames)

        return embeddings

class TextEncoder (Module):
    def __init__ (self, num_layers, dropout, hidden_dim, emb_dim, emb_layer, device):
        super().__init__()

        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.embedding_dim = emb_dim
        self.word_embeddings = emb_layer
        self.device = device

        self.lstm = LSTM(self.embedding_dim, self.hidden_dim, self.num_layers)

        self.initialise_weights ()
        
    def forward (self, text, hidden):
        embeds = self.word_embeddings(text.view (1, -1))
        
        lstm_out, hidden = self.lstm(embeds.view(embeds.shape [1], 1, -1), hidden)

        return lstm_out, hidden
    
    def initialise_weights (self):
        for param in self.lstm.parameters():
            if len(param.shape) >= 2:
                orthogonal_(param.data)
            else:
                normal_(param.data)
            
    def init_state(self, batch_sz):
        return (torch.zeros(self.num_layers, batch_sz, self.hidden_dim, device=self.device),
                torch.zeros(self.num_layers, batch_sz, self.hidden_dim, device=self.device))

class AudioVideoEncoder (Module):
    def __init__(self, download_pretrained=False):
        super().__init__()

        self.audio_enc = AudioEncoder ()
        self.video_enc = VideoEncoder (download_pretrained)

    def forward (self, audio_file, video_frames):
        audio_out = self.audio_enc (audio_file)
        audio_emb = audio_out.view (1, -1)
        # print (audio_emb.shape)

        video_emb = self.video_enc (video_frames)
        # print (video_emb.shape)

        enc_output = torch.cat ((audio_emb, video_emb), dim=1)

        return enc_output
