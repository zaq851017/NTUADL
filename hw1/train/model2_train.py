#!/usr/bin/env python
# coding: utf-8

# In[22]:


import argparse
import logging
import os
import json
import pickle
from pathlib import Path
from utils import Tokenizer, Embedding
from dataset import Seq2SeqDataset
from tqdm import tqdm
import torch
from torch import nn
from torch.autograd import Variable as V
import numpy as np


# In[23]:


with open('../datasets/seq2seq/train.pkl', 'rb') as file:
    train =pickle.load(file)
with open('../datasets/seq2seq/valid.pkl', 'rb') as file:
    valid =pickle.load(file)
with open('../datasets/seq2seq/embedding.pkl', 'rb') as file:
    embed = pickle.load(file)


# In[24]:


BATCH_SIZE = 9
WORD_VECTOR_LENGTH = 300
EPOCH = 3
hidden_size = 50


# In[25]:


#collate_fn=train.collate_fn
loader = torch.utils.data.DataLoader(dataset=train, batch_size=BATCH_SIZE, shuffle=False,collate_fn=train.collate_fn)


# In[26]:


class ENCODER(nn.Module):
    def __init__(self):
        super(ENCODER, self).__init__()
        embedding_weight = embed.vectors
        self.embedding = nn.Embedding.from_pretrained(embedding_weight)
        self.encoder = nn.GRU(         # if use nn.RNN(), it hardly learns
            input_size=WORD_VECTOR_LENGTH,
            hidden_size=50,         # rnn hidden unit
            num_layers=3,           # number of rnn layer
            batch_first=True,
            bidirectional=False
        )
        self.out = nn.Linear(50, 50)
        self.func = nn.Tanh()
        
    #h_n shape (n_layers, batch, hidden_size)
    #h_n(6,10,50)
    def forward(self, x):
        embed_vector = self.embedding(x)
        output, hidden = self.encoder(embed_vector)
        hidden= self.out(hidden)
        hidden = self.func(hidden)
        #out = self.out(h_n[::,-1])
        #return torch.tanh(out)
        return hidden


encoder = ENCODER()
encoder.cuda()
print(encoder)


# In[27]:


class DECODER(nn.Module):
    def __init__(self):
        super(DECODER, self).__init__()
        #self.output_dim = embed.vectors.shape[0]
        embedding_weight = embed.vectors
        self.embedding = nn.Embedding.from_pretrained(embedding_weight)
        self.decoder = nn.GRU(         # if use nn.RNN(), it hardly learns
            input_size=WORD_VECTOR_LENGTH,
            hidden_size=50,         # rnn hidden unit
            num_layers=3,           # number of rnn layer
            batch_first=True,
            bidirectional=False
        )
        #self.out = nn.Linear(hidden_size*2,embedding_weight.shape[0] )
        self.out = nn.Linear(hidden_size, embedding_weight.shape[0])
        #self.softmax = nn.Softmax(dim=2)
    #h_n shape (n_layers, batch, hidden_size)
    def forward(self, x ,hidden):
        out = self.embedding(x)
        out, hidden = self.decoder(out,hidden)
        out = self.out(out)
        #out = self.softmax(out)
        #return self.out(out)
        return out,hidden
   


decoder = DECODER()
decoder.cuda()
print(decoder)


# In[28]:


LR = 0.001 
opt1 = torch.optim.Adam(encoder.parameters(), lr=LR)
opt2 = torch.optim.Adam(decoder.parameters(), lr=LR)
loss_func = torch.nn.CrossEntropyLoss()  


# In[29]:


for epoch in range(EPOCH):
    tmp_lo=0
    num = 0
    for i,train_data in enumerate(tqdm(loader)):
        tmp = train_data['text'][0]
        tmp2 = train_data['summary'][0]
        btach_num = train_data['text'].shape[0]
        predict_len =train_data['summary'].shape[1]
        vocab_len = embed.vectors.shape[0]
        #將ans對齊
        ans = train_data['summary'][:,1:predict_len]
        zero_ten = torch.zeros(btach_num,1,dtype=torch.long)
        ans = torch.cat((ans,zero_ten),dim=1)
        #
        for j in range(1,btach_num):
            b = train_data['text'][j]
            c = train_data['summary'][j]
            tmp=torch.cat( (tmp,b) ,dim = 0)
            tmp2=torch.cat( (tmp2,c) ,dim = 0)
        tmp = tmp.view(btach_num,-1)
        tmp2 = tmp2.view(btach_num,-1)
        tmp = V(tmp).cuda()
        tmp2 = V(tmp2).cuda()
        encode_output = encoder(tmp)
        hidden = encode_output
        outputs = torch.zeros(btach_num , tmp2.shape[1], embed.vectors.shape[0]).cuda()
        for j in range(tmp2.shape[1]):
            ls=tmp2[:,j].unsqueeze(-1)
            decode_output,hidden = decoder(ls,hidden)
            decode_output = torch.squeeze(decode_output,1)
            outputs[:,j,:] = decode_output
            #print(decode_output.shape)
        #outputs = outputs.view(btach_num,-1)
        #ans_embed = ans_embed.view(btach_num,-1)
        ans = ans.cuda()
        outputs = outputs.view((-1,vocab_len))
        ans = ans.view(-1)
        loss = loss_func(outputs,ans)
        opt1.zero_grad()
        opt2.zero_grad()
        loss.backward() 
        opt1.step()
        opt2.step()
        num = num +1
        tmp_lo = tmp_lo+loss
    print(tmp_lo/num)
    torch.save(encoder, 'ab_tanh_encoder'+str(epoch)+'.pkl')
    torch.save(decoder, 'ab_tanh_decoder'+str(epoch)+'.pkl')

