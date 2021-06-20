import torch
from torch.utils.data import Dataset, DataLoader

import os
import numpy as np
import json

class VQGDataset (Dataset):
    def __init__ (self, questions_file, vocab_file, idx_2_word_file, frames_path, audio_path, text_transform=None, video_transform=None):
        with open (questions_file, 'r') as file_io:
            self.questions = json.load (file_io)
        
        with open (vocab_file, 'r') as file_io:
            self.vocab = json.load (file_io)
        
        with open (idx_2_word_file, 'r') as file_io:
            self.index_to_word = json.load (file_io)

        self.frames_path = frames_path
        self.audio_path = audio_path
        self.text_transform = text_transform
        self.video_transform = video_transform

    def __len__ (self):
        return len (self.questions)
    
    def __getitem__ (self, idx):
        video_id = self.questions [idx] ['video_id']
        question_id = self.questions [idx] ['question_id']
        context_str = self.questions [idx] ['context']
        question_str = self.questions [idx] ['question']
        answer_str = self.questions [idx] ['answer']

        # Text
        if self.text_transform:
            context_tensor = self.text_transform (f'{answer_str} <sep> {context_str}', self.vocab)

        # Video
        frames = torch.from_numpy (np.load (os.path.join (self.frames_path, f'v_{video_id}_q_{question_id}_.npy')))
        if self.video_transform:
            frames = self.video_transform (frames)

        # Audio
        audio_file = os.path.join (self.audio_path, f'v_{video_id}_q_{question_id}_.wav')

        # Target Question
        # if self.text_transform:
        #     question = self.text_transform (f"{question_str}", self.vocab)
        
        if self.text_transform:
            target = self.text_transform (f"{question_str} <end>", self.vocab)

        context_seq_len = context_tensor.shape [0]
        target_seq_len = target.shape [0]
        
        return frames, audio_file, context_tensor, question_id, question_str, target, context_seq_len, target_seq_len


class VQGCharDataset (Dataset):
    def __init__ (self, questions_file, vocab_file, idx_2_word_file, frames_path, audio_path, text_transform=None, video_transform=None):
        with open (questions_file, 'r') as file_io:
            self.questions = json.load (file_io)
        
        with open (vocab_file, 'r') as file_io:
            self.vocab = json.load (file_io)
        
        with open (idx_2_word_file, 'r') as file_io:
            self.index_to_word = json.load (file_io)

        self.frames_path = frames_path
        self.audio_path = audio_path
        self.text_transform = text_transform
        self.video_transform = video_transform

    def __len__ (self):
        return len (self.questions)
    
    def __getitem__ (self, idx):
        video_id = self.questions [idx] ['video_id']
        question_id = self.questions [idx] ['question_id']
        context_str = self.questions [idx] ['context']
        answer_str = self.questions [idx] ['answer']
        question_str = self.questions [idx] ['question']

        # Text
        if self.text_transform:
            context_tensor = self.text_transform (f'{answer_str} <sep> {context_str}', self.vocab)

        # Video
        frames = torch.from_numpy (np.load (os.path.join (self.frames_path, f'v_{video_id}_q_{question_id}_.npy')))
        if self.video_transform:
            frames = self.video_transform (frames)

        # Audio
        audio_file = os.path.join (self.audio_path, f'v_{video_id}_q_{question_id}_.wav')

        # Target Question
        # if self.text_transform:
        #     question = self.text_transform (f"{question_str}", self.vocab)
        
        if self.text_transform:
            target = self.text_transform (f"{question_str} <end>", self.vocab)

        context_seq_len = context_tensor.shape [0]
        target_seq_len = target.shape [0]
        
        return frames, audio_file, context_tensor, question_id, question_str, target, context_seq_len, target_seq_len

class SquadDataset (Dataset):
    def __init__ (self, questions_file, vocab_file, idx_2_word_file, text_transform=None):
        with open (questions_file, 'r') as file_io:
            self.questions = json.load (file_io)
        
        with open (vocab_file, 'r') as file_io:
            self.vocab = json.load (file_io)
        
        with open (idx_2_word_file, 'r') as file_io:
            self.index_to_word = json.load (file_io)

        self.text_transform = text_transform

    def __len__ (self):
        return len (self.questions)
    
    def __getitem__ (self, idx):
        context_str = self.questions [idx] ['context']
        question_str = self.questions [idx] ['question']
        answer_str = self.questions [idx] ['answer']

        # print (f'answer - {answer_str}')
        # print (f'context - {context_str}')
        # print (f'ques - {question_str}')

        if self.text_transform:
            context_tensor = self.text_transform (f'{answer_str} <sep> {context_str}', self.vocab)
        
        if self.text_transform:
            target = self.text_transform (f"{question_str} <end>", self.vocab)

        context_seq_len = context_tensor.shape [0]
        target_seq_len = target.shape [0]
        
        return context_tensor, question_str, target, context_seq_len, target_seq_len

# def handle_special_tokens (text, wtoi):
#     text = text.replace ('<start>', f"{wtoi ['<start>']}")
#     text = text.replace ('<end>', f"{wtoi ['<end>']}")
#     text = text.replace ('<sep>', f"{wtoi ['<sep>']}")
#     text = text.replace ('<pad>', f"{wtoi ['<pad>']}")
#     text = text.replace ('<unk>', f"{wtoi ['<unk>']}")
#     return text

# def is_special_tokens (token):
#     if token == '<start>' or token == '<end>' or token == '<pad>' or token == '<unk>' or token == '<sep>':
#         return True
#     return False

# def prepare_char_seq (text, wtoi):
#     tokens = text.split (' ')
#     ids = []
#     for tok in tokens:
#         if is_special_tokens (tok):
#             # print (f'{tok} - {wtoi [tok]}')
#             ids.append (wtoi [tok])
#         else:
#             for char in tok:
#                 # print (f'{char} - {wtoi [char]}')
#                 ids.append (wtoi [char])
#     return torch.tensor(ids, dtype=torch.long)


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

# if __name__ == '__main__':
#     train_dataset = SquadDataset ('squad/prep_val.json', 'data/char_vocab.json', 'data/index_to_char.json', prepare_char_seq)
#     train_dataloader = DataLoader (train_dataset, batch_size=1, shuffle=False)

#     for context_tensor, question_str, target, context_seq_len, target_seq_len in train_dataloader:
#         # print (f'question - {q}')
#         print (f'context - {context_tensor.shape}')
#         print (f'target - {target.shape}')
#         print (f'context len - {context_seq_len}')
#         print (f'target len - {target_seq_len}')
#         break