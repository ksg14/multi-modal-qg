import torch
from torch.nn import Module, LSTM, Linear, Dropout, GRU, Embedding, Conv2d, MaxPool2d, AdaptiveAvgPool1d, BatchNorm2d, Flatten
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_, orthogonal_, normal_

import torchvision.models as models

from transformers import ProphetNetEncoder

class AudioEncoder (Module):
    def __init__ (self):
        super().__init__()
        
        self.vggish = torch.hub.load('harritaylor/torchvggish', 'vggish', postprocess=False)
        self.adapt_avg_pool = AdaptiveAvgPool1d(1)
        # self.fc1 = Linear (128, audio_emb) 

    def forward (self, audio_file):
        out = self.vggish.forward (audio_file)
        # embeddings = self.adapt_avg_pool (out.view (1, out.shape [1], -1))
        return out

class VideoResnetEncoder (Module):
    def __init__ (self, download_pretrained=False):
        super().__init__()
        self.resnet3d = models.video.r2plus1d_18 (pretrained=download_pretrained, progress=True)

    def forward (self, video_frames):
        embeddings = self.resnet3d (video_frames)

        return embeddings

class VideoConvLstmEncoder (Module):
    def __init__ (self, in_channels, kernel_sz, stride, hidden_dim, video_emb_dim):
        super().__init__()
        self.in_channels = in_channels
        self.kernel_sz = kernel_sz
        self.stride = stride
        self.hidden_dim = hidden_dim
        self.video_emb_dim = video_emb_dim

        self.conv1 = Conv2d (self.in_channels, 4, self.kernel_sz, self.stride)
        self.bn1 = BatchNorm2d (4)
        self.conv2 = Conv2d (4, 6, self.kernel_sz, self.stride)
        self.bn2 = BatchNorm2d (6)
        self.maxpool1 = MaxPool2d (self.kernel_sz, self.kernel_sz)

        self.conv3 = Conv2d (6, 8, self.kernel_sz, self.stride)
        self.bn3 = BatchNorm2d (8)
        self.conv4 = Conv2d (8, 10, self.kernel_sz, self.stride)
        self.bn4 = BatchNorm2d (10)
        self.maxpool2 = MaxPool2d (self.kernel_sz, self.kernel_sz)

        self.flatten = Flatten ()

        self.lstm = LSTM(self.video_emb_dim, self.hidden_dim)

        self.initialise_weights ()

    def forward (self, video_frames):
        batch_sz = video_frames.shape [2]
        channels = video_frames.shape [1]
        height = video_frames.shape [3]
        width = video_frames.shape [4]

        first_block = self.maxpool1 (self.bn2 (F.relu (self.conv2 (self.bn1 (F.relu (self.conv1 (video_frames.view (batch_sz, channels, height, width))))))))
        second_block = self.maxpool2 (self.bn4 (F.relu (self.conv4 (self.bn3 (F.relu (self.conv3 (first_block)))))))

        cnn_out = self.flatten (second_block)

        lstm_out, _ = self.lstm (cnn_out.view (cnn_out.shape [0], 1, -1))

        return lstm_out
    
    def initialise_weights (self):
        for param in self.lstm.parameters():
            if len(param.shape) >= 2:
                orthogonal_(param.data)
            else:
                normal_(param.data)

class TextEncoder (Module):
    def __init__ (self, num_layers, dropout_p, hidden_dim, emb_dim, emb_layer, device):
        super().__init__()

        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.embedding_dim = emb_dim
        self.word_embeddings = emb_layer
        self.device = device
        self.dropout_p = dropout_p

        self.lstm = LSTM(self.embedding_dim, self.hidden_dim, self.num_layers, dropout=self.dropout_p)

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
    def __init__(self, av_in_channels, av_kernel_sz, av_stride, av_hidden_dim, video_emb_dim):
        super().__init__()

        self.audio_enc = AudioEncoder ()
        # self.video_enc = VideoEncoder (download_pretrained)
        self.video_enc = VideoConvLstmEncoder (av_in_channels, av_kernel_sz, av_stride, av_hidden_dim, video_emb_dim)

    def forward (self, audio_file, video_frames):
        audio_out = self.audio_enc (audio_file)
        audio_emb = audio_out.view (1, -1)
        # print (audio_emb.shape)

        video_emb = self.video_enc (video_frames).squeeze ()
        # print (f'video emb shape - {video_emb.shape}')

        # enc_output = torch.cat ((audio_emb, video_emb), dim=1)

        return audio_emb, video_emb

class ProphetNetTextEncoder (Module):
    def __init__(self, enc_path):
        super().__init__()

        self.encoder = ProphetNetEncoder.from_pretrained (enc_path)
    
    def forward (self):
        pass
