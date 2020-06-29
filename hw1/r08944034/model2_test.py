#!/usr/bin/env python
# coding: utf-8

# In[31]:


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
from argparse import ArgumentParser


# In[32]:


encode_path='model2_encoder.pkl'
decode_path='model2_decoder.pkl'
#output_path='batch_valid.jsonl'
parser = ArgumentParser()
parser.add_argument('--output_path')
args  = parser.parse_args()


# In[33]:


with open('./data.pkl', 'rb') as file:
    valid =pickle.load(file)
with open('./embedding.pkl', 'rb') as file:
    embed = pickle.load(file)


# In[34]:


BATCH_SIZE = 10
WORD_VECTOR_LENGTH = 300
hidden_size = 50


# In[35]:


loader = torch.utils.data.DataLoader(dataset=valid, batch_size=BATCH_SIZE, shuffle=False,collate_fn=valid.collate_fn)


# In[36]:


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
        #self.out = nn.Linear(300, 1)
    #h_n shape (n_layers, batch, hidden_size)
    #h_n(6,10,50)
    def forward(self, x):
        embed_vector = self.embedding(x)
        output, hidden = self.encoder(embed_vector)
        #out = self.out(h_n[::,-1])
        #return torch.tanh(out)
        return hidden


# In[37]:


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


# In[38]:


encoder = torch.load(encode_path)
decoder = torch.load(decode_path)
encoder.cuda()
decoder.cuda()
print(encoder)
print(decoder)


# In[39]:


f = open(args.output_path,mode='w')
for i,input_data in enumerate(tqdm(loader)):
    batch_num = input_data['text'].shape[0]
    input_text = input_data['text']
    seq_len = input_data['text'].shape[1]
    input_text =V(input_text).cuda()
    start_out = torch.zeros([batch_num,1],dtype=torch.long).cuda()
    for m in range(batch_num):
        start_out[m] = input_data['text'][m][0]
    start_out = torch.unsqueeze(start_out,-1)
    encoder_output = encoder(input_text)
    for m in range(batch_num):
        hidden = encoder_output[:,m,:]
        hidden = torch.unsqueeze(hidden,1).contiguous()
        out_words=[]
        for j in range(80):
            if j==0:
                output_decoder,hidden = decoder(start_out[m],hidden)
                hidden = hidden.cuda()
                max_prob,max_index  = torch.max(output_decoder,2)
                output_words = embed.vocab[max_index.item()]
                output_index = max_index.item()
            else:
                input_vocab = torch.tensor([output_index])
                input_vocab = input_vocab.unsqueeze(-1)
                input_vocab = input_vocab.cuda()
                output_decoder,hidden = decoder(input_vocab,hidden)
            
                max_prob,max_index  = torch.max(output_decoder,2)
                output_words = embed.vocab[max_index.item()]
                output_index = max_index.item()
        
            
            if output_index==2:
                break
            out_words.append(output_words)
            #print(output_words)
        
        out_str = " ".join(out_words)
        ans_list={}
        ans_list['id'] = input_data['id'][m]
        ans_list['predict'] = out_str
        f.write(json.dumps(ans_list) + "\n")
        #print(ans_list)
    
    
    #print(i)
    #f.write(json.dumps(ans_list) + "\n")
    
    
f.close()


# In[ ]:



# In[ ]:




