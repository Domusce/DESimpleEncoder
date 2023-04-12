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

        # nn.init.xavier_uniform(self.ent_encoder.weight)

    
    def create_time_embedds(self):
        
        # frequency embeddings for the entities
        self.m_freq_h = nn.Embedding(self.numEnt, self.t_emb_dim).cuda()
        self.m_freq_t = nn.Embedding(self.numEnt, self.t_emb_dim).cuda()
        
        self.d_freq_h = nn.Embedding(self.numEnt, self.t_emb_dim).cuda()
        self.d_freq_t = nn.Embedding(self.numEnt, self.t_emb_dim).cuda()

        self.y_freq_h = nn.Embedding(self.numEnt, self.t_emb_dim).cuda()
        self.y_freq_t = nn.Embedding(self.numEnt, self.t_emb_dim).cuda()

        # phi embeddings for the entities 
        self.m_phi_h = nn.Embedding(self.numEnt, self.t_emb_dim).cuda()
        self.m_phi_t = nn.Embedding(self.numEnt, self.t_emb_dim).cuda()
        
        self.d_phi_h = nn.Embedding(self.numEnt, self.t_emb_dim).cuda()
        self.d_phi_t = nn.Embedding(self.numEnt, self.t_emb_dim).cuda()

        self.y_phi_h = nn.Embedding(self.numEnt, self.t_emb_dim).cuda()
        self.y_phi_t = nn.Embedding(self.numEnt, self.t_emb_dim).cuda()

        # amps embeddings for the entities
        self.m_amps_h = nn.Embedding(self.numEnt, self.t_emb_dim).cuda()
        self.m_amps_t = nn.Embedding(self.numEnt, self.t_emb_dim).cuda()
        
        self.d_amps_h = nn.Embedding(self.numEnt, self.t_emb_dim).cuda()
        self.d_amps_t = nn.Embedding(self.numEnt, self.t_emb_dim).cuda()

        self.y_amps_h = nn.Embedding(self.numEnt, self.t_emb_dim).cuda()
        self.y_amps_t = nn.Embedding(self.numEnt, self.t_emb_dim).cuda()


        nn.init.xavier_uniform_(self.m_freq_h.weight)
        nn.init.xavier_uniform_(self.d_freq_h.weight)
        nn.init.xavier_uniform_(self.y_freq_h.weight)
        nn.init.xavier_uniform_(self.m_freq_t.weight)
        nn.init.xavier_uniform_(self.d_freq_t.weight)
        nn.init.xavier_uniform_(self.y_freq_t.weight)

        nn.init.xavier_uniform_(self.m_phi_h.weight)
        nn.init.xavier_uniform_(self.d_phi_h.weight)
        nn.init.xavier_uniform_(self.y_phi_h.weight)
        nn.init.xavier_uniform_(self.m_phi_t.weight)
        nn.init.xavier_uniform_(self.d_phi_t.weight)
        nn.init.xavier_uniform_(self.y_phi_t.weight)

        nn.init.xavier_uniform_(self.m_amps_h.weight)
        nn.init.xavier_uniform_(self.d_amps_h.weight)
        nn.init.xavier_uniform_(self.y_amps_h.weight)
        nn.init.xavier_uniform_(self.m_amps_t.weight)
        nn.init.xavier_uniform_(self.d_amps_t.weight)
        nn.init.xavier_uniform_(self.y_amps_t.weight)

