import torch
from torch.utils.data import Dataset, DataLoader

import os
import numpy as np
import json

class VQGDataset (Dataset):
    def __init__ (self, questions_file, vocab_file, idx_2_word_file, frames_path, audio_path, text_transform=None, prophetnet_transform=None, video_transform=None):
        with open (questions_file, 'r') as file_io:
            self.questions = json.load (file_io)
        
        with open (vocab_file, 'r') as file_io:
            self.vocab = json.load (file_io)
        
        with open (idx_2_word_file, 'r') as file_io:
            self.index_to_word = json.load (file_io)

        self.frames_path = frames_path
        self.audio_path = audio_path
        self.text_transform = text_transform
        self.prophetnet_transform = prophetnet_transform
        self.video_transform = video_transform

    def __len__ (self):
        return len (self.questions)
    
    def __getitem__ (self, idx):
        video_id = self.questions [idx] ['video_id']
        question_id = self.questions [idx] ['question_id']
        context_str = self.questions [idx] ['context']
        question_str = self.questions [idx] ['question']
        answer_str = self.questions [idx] ['answer']

        # Video
        frames = torch.from_numpy (np.load (os.path.join (self.frames_path, f'v_{video_id}_q_{question_id}_.npy')))
        if self.video_transform:
            frames = self.video_transform (frames)

        # Audio
        audio_file = os.path.join (self.audio_path, f'v_{video_id}_q_{question_id}_.wav')

        # Text for LSTM
        if self.text_transform:
            context_tensor = self.text_transform (context_str, self.vocab)        
            target = self.text_transform (f"{question_str} <end>", self.vocab)

            context_seq_len = context_tensor.shape [0]
            target_seq_len = target.shape [0]
        
            return frames, audio_file, context_tensor, question_id, question_str, target, context_seq_len, target_seq_len

        # Text for ProphetNet
        if self.prophetnet_transform:
            question = self.prophetnet_transform (f'{self.prophetnet_transform.cls_token} {question_str}', return_tensors='pt')

            # print (f'q ids - {question.input_ids}')
            # print (f'start - {self.prophetnet_transform.decode (question.input_ids [0][0])}')
            # print (f'end - {self.prophetnet_transform.decode (question.input_ids [0][-1])}')

            question_src = question.input_ids [:, :-1]
            question_tgt = question.input_ids [:, 1:]

            context = self.prophetnet_transform (f'{answer_str} [SEP] {context_str}', return_tensors='pt')

            return frames, audio_file, context.input_ids, question_src, question_tgt, question_id, question_str

# if __name__ == '__main__':
#     train_dataset = VQGDataset (config.train_file, vocab_file, config.salient_frames_path, config.salient_audio_path, prepare_sequence)
#     train_dataloader = DataLoader (train_dataset, batch_size=1, shuffle=False)

#     for _, (frames, audio_file, context_tensor, target, context_len, target_len) in enumerate (train_dataloader):
#         # print (f'question - {q}')
#         print (f'frame - {frames.shape}')
#         print (f'audio - {audio_file}')
#         print (f'context - {context_tensor.shape}')
#         print (f'target - {target.shape}')
#         print (f'context len - {context_len}')
#         print (f'target len - {target_len}')
#         break