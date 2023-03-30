import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence as pack
import numpy as np
import spacy

Glove_path = "glove/glove.6B.300d.txt"

class DESimplE_GRUEcoder(nn.Module):
    def __init__(self, word2id, tokenizer,se_prop=0, hidden_size=100, pretrained_vocab=None, pretrained_emb=None):
        super(DESimplE_GRUEcoder, self).__init__()

        self.word2id = word2id
        self.tokenizer = tokenizer
        self.se_prop = se_prop
        self.init_embeddings(pretrained_vocab,pretrained_emb)
        self.hidden_size = hidden_size
        
        self.numEmb = self.embed_matrix.shape[0]
        self.emb_dim = self.embed_matrix.shape[1]
        self.temporal_emb_dim = self.embed_matrix.shape[1] - int(self.se_prop*self.embed_matrix.shape[1])
        self.embed = nn.Embedding(num_embeddings=self.numEmb,
                                  embedding_dim=self.emb_dim + self.temporal_emb_dim,
                                  padding_idx=0)
        
        self.encoder = nn.GRU(self.emb_dim+self.temporal_emb_dim, self.hidden_size, batch_first=True)




