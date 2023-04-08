import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from bisect import bisect_left

import encoder
import random
import tqdm

class DE_SimplE(nn.Module):
    def __init__(self, data, ent_encoder, rel_encoder, numEnt, s_emb_dim, t_emb_dim, dimenson=300, dropout=0.1):
        super(DE_SimplE).__init__()

        self.data = data
        self.numEnt = numEnt
        self.dropout = dropout
        self.dimenson = dimenson
        self.s_emb_dim = s_emb_dim
        self.t_emb_dim = t_emb_dim
        self.ent_encoder = ent_encoder
        self.rel_encoder = rel_encoder

        self.time_nl = torch.sin

        nn.init.xavier_uniform