import json
from pathlib import Path
import os
import numpy as np
from tqdm import tqdm

def get_maxlen_text (corpus, key):
    max_len = 0
    avg_len = 0
    for question in tqdm (corpus):
        n_tokens = len (question [key].split ())
        max_len = max (max_len, n_tokens)
        avg_len += n_tokens
    return max_len, avg_len / len (corpus)

def get_maxlen_frames (path):
    max_len = 0
    avg_len = 0
    for frames_file in tqdm (os.listdir (path)):
        frames = np.load (os.path.join (path, frames_file))
        max_len = max (max_len, frames.shape [0])
        avg_len += frames.shape [0]
    return max_len, avg_len / len (os.listdir (path))

if __name__ == '__main__':
    dataset_path = Path (r'dataset')
    questions_file = 'data/preprocesses_text.json'
    frames_path = dataset_path / 'salient_frames'

    with open (questions_file, 'r') as file_io:
        questions = json.load (file_io)

    print (f"Max/Avg len context - {get_maxlen_text (questions, 'context')}")
    print (f"Max/Avg len question - {get_maxlen_text (questions, 'question')}")
    print (f"Max/Avg len frames - {get_maxlen_frames (frames_path)}")



