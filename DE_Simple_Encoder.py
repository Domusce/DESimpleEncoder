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

        self.embed.weight.data.copy_(torch.from_numpy(self.embed_matrix))
        self.tokenization_memo={}

    
    def init_embedding(self, pretrained_vocab, pretrained_emb):
        self.word_embed={}
        if pretrained_vocab is None or pretrained_emb is None:
            with open(Glove_path, encoding='utf-8') as glove:
                for line in glove:
                    word, vec = line.split(' ', 1)
                    if word in self.word2id:
                        self.word_embed[self.word2id[word]] = np.fromstring(vec,sep=' ')
        
        else:
            for word, w_id in pretrained_vocab:
                if word in self.word2id:
                    self.word_embed[self.word2id[word]] = pretrained_emb[w_id]


        # initialize unknown word according to normal distribution
        uninitialized = [word for word in self.word2id.values() if not word in self.word_embed]
        for word in uninitialized:
            self.word_embed[word] = np.random.normal(size=300)
        
        self.embed_matix = np.zeros((len(self.word_embed),300))
        for word in self.word_embed:
            self.embed_matix[word] = self.word_embed[word]

    
    def forward(self, batch, doc_len):
        size, sort = torch.sort(doc_len, dim=0, descending=True)
        _, unsort = torch.sort(sort, dim=0)
        batch = torch.index_select(batch, dim=0, index=sort)
        embedded = self.embed(batch)
        packed = pack(embedded, size.data.tolist(), batch_first=True)
        encoded, h = self.encoder(packed)
        return torch.index_select(h, dim=1, index=unsort)[0]