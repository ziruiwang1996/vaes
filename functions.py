import numpy as np
import torch

def one_hot(seq_list):
    tokens = ["A", "C", "D", "E", "F", "G", "H", "I", "K", "L", "M", "N", "P", "Q", "R", "S", "T", "V", "W", "Y"]
    order = np.arange(len(tokens))
    look_up = dict(zip(tokens, order))
    emb = []
    for seq in seq_list:
        one_hot_emb = np.zeros((58, 20))  # dimension: seq_len * d_model
        n = 0
        for aa in list(seq):
            one_hot_emb[n][look_up.get(aa)] = 1
            n += 1
        emb.append(one_hot_emb)
    emb = torch.from_numpy(np.array(emb))
    return emb