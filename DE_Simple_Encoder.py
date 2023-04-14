import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence as pack
import numpy as np
import spacy

Glove_path = "glove/glove.6B.300d.txt"

class DESimplE_GRUEcoder(nn.Module):
    def __init__(self, word2id, tokenizer,se_prop=0, E_or_R = 0 ,hidden_size=100, pretrained_vocab=None, pretrained_emb=None):
        super(DESimplE_GRUEcoder, self).__init__()

        self.word2id = word2id
        self.tokenizer = tokenizer
        self.se_prop = se_prop
        self.E_or_R = E_or_R
        self.init_embeddings(pretrained_vocab,pretrained_emb)
        self.hidden_size = hidden_size
        
        self.numEnt = self.embed_matrix.shape[0]
        self.emb_dim = self.embed_matrix.shape[1]
        self.s_emb_dim = int(self.se_prop*self.emb_dim)
        self.t_emb_dim = self.emb_dim - self.s_emb_dim
        self.embed = nn.Embedding(num_embeddings=self.numEnt,
                                  embedding_dim=self.s_emb_dim + self.E_or_R*self.t_emb_dim,
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

    def prepare_batch(self, texts):
        """texts
        Gets a list of untokenized texts,
        :param texts:
        :return: a padded batch that can be used  as input to forward,
        puts it on the same device as the GRU
        """
        batch_size = len(texts)
        tokenized_texts = []
        for text in texts:
            if text in self.tokenization_memo:
                tokenized_texts.append(self.tokenization_memo[text])
            else:
                tokenized = [self.word2id[token.text.lower()]\
                             for token in self.tokenizer(text,
                                                         disable=['parser','tagger','ner'])\
                             if token.text.lower() in self.word2id]
                self.tokenization_memo[text] = tokenized
                tokenized_texts.append(tokenized)

        padded_length = max(len(text) for text in tokenized_texts)
        phrase_batch = np.zeros((batch_size, padded_length),dtype=int)

        for i, tokens in enumerate(tokenized_texts):
            phrase_batch[i,0:len(tokens)] = np.arrays(tokens)

        device = self.encoder.weight_ih_10.device
        phrase_batch = torch.from_numpy(phrase_batch).to(device)
        phrase_len = torch.LongTensor([len(text) for text in tokenized_texts]).to(device)
        return  phrase_batch, phrase_len

    def get_Data_numEnt(self):
        return self.numEnt

    def get_Data_Dimenson(self):
        return self.s_emb_dim, self.t_emb_dim
