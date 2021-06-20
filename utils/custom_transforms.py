import torch

def get_word_from_idx (idx, itow):
    return itow [idx]

def resize (vid, size, interpolation='bicubic'):
    scale = None
    if isinstance(size, int):
        scale = float(size) / min(vid.shape[-2:])
        size = None
    return torch.nn.functional.interpolate(
        vid, size=size, scale_factor=scale, mode=interpolation, align_corners=False)

def to_normalized_float_tensor(vid):
    return vid.permute(3, 0, 1, 2).to(torch.float32) / 255

def normalize(vid, mean, std):
    shape = (-1,) + (1,) * (vid.dim() - 1)
    mean = torch.as_tensor(mean).reshape(shape)
    std = torch.as_tensor(std).reshape(shape)
    return (vid - mean) / std

def prepare_sequence (seq, to_ix):
    idxs = [to_ix[w] for w in seq.split ()]
    return torch.tensor(idxs, dtype=torch.long)

def is_special_tokens (token):
    if token == '<start>' or token == '<end>' or token == '<pad>' or token == '<unk>' or token == '<sep>':
        return True
    return False

def prepare_char_seq (text, wtoi):
    tokens = text.split (' ')
    ids = []
    for tok in tokens:
        if is_special_tokens (tok):
            # print (f'{tok} - {wtoi [tok]}')
            ids.append (wtoi [tok])
        else:
            for char in tok:
                # print (f'{char} - {wtoi [char]}')
                ids.append (wtoi [char])
    return torch.tensor(ids, dtype=torch.long)

class Resize(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, vid):
        return resize(vid, self.size)

class ToFloatTensor(object):
    def __call__(self, vid):
        return to_normalized_float_tensor(vid)

class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, vid):
        return normalize(vid, self.mean, self.std)
