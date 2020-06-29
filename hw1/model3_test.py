#!/usr/bin/env python
# coding: utf-8

# In[1]:


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
import torch.nn.functional as F
from argparse import ArgumentParser


# In[2]:


encode_path='model3_encoder.pkl'
decode_path='model3_decoder.pkl'
parser = ArgumentParser()
parser.add_argument('--output_path')
args  = parser.parse_args()


# In[3]:


with open('./data.pkl', 'rb') as file:
    valid =pickle.load(file)
with open('./embedding.pkl', 'rb') as file:
    embed = pickle.load(file)



# In[4]:


BATCH_SIZE = 10
WORD_VECTOR_LENGTH = 300
hidden_size = 150
num_layers=2
vocab_size = embed.vectors.shape[0]
b_dir = 2



# In[5]:


valid_loader = torch.utils.data.DataLoader(dataset=valid, batch_size=BATCH_SIZE, shuffle=False,collate_fn=valid.collate_fn)


# In[6]:


class ENCODER(nn.Module):
    def __init__(self):
        super(ENCODER, self).__init__()
        embedding_weight = embed.vectors
        self.embedding = nn.Embedding.from_pretrained(embedding_weight)
        self.rnn = nn.GRU(WORD_VECTOR_LENGTH, hidden_size, num_layers,batch_first=True, bidirectional=True)
    def forward(self, x):
        embed_vector = self.embedding(x)
        output, hidden = self.rnn(embed_vector)
        #out = self.out(h_n[::,-1])
        #return torch.tanh(out)
        return output,hidden

# In[13]:


class AttnDecoderRNN(nn.Module):
    def __init__(self):
        super(AttnDecoderRNN, self).__init__()
        embedding_weight = embed.vectors
        self.embedding = nn.Embedding.from_pretrained(embedding_weight)
        self.attn = nn.Linear((WORD_VECTOR_LENGTH+hidden_size*num_layers*b_dir), 300)
        self.rnn = nn.GRU(600, hidden_size, 2, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(300, vocab_size)

    def forward(self, input, hidden, encoder_output,batch_num):
        # encode_output.shape = [9, 299, 300] = [batch , input_len , hidden*dir]
        # encode_hidden = [4, 9, 150] = [num_layer , batch,hidden ]
        # encode_output.shape = [9, 299, 150]
        embed_vector = self.embedding(input) #embed_vector=[9, 1, 300]
        tmp_hidden = hidden
        tmp_hidden = tmp_hidden.transpose(0, 1) # hidden = [9, 6, 50]
        tmp_hidden = tmp_hidden.contiguous().view(1,1,-1) # hidden = [9, 1, 600]
        tt = torch.cat( (embed_vector,tmp_hidden),2 )
        tt = self.attn(tt)
        attn_weights = F.softmax(tt, dim=2) #attn_weights = ([9, 1, 300])
        attn_weights = attn_weights.narrow(2, 0, encoder_output.shape[1]) #attn_weights = ([9, 1, 299])
        attn_applied = torch.bmm(attn_weights , encoder_output )  # attn_applied =[9, 1, 300]
        attn_embed = torch.cat((attn_applied,embed_vector),2 )
        output,hidden = self.rnn(attn_embed,hidden)
        output = self.fc(output)
        return output,hidden
        


# In[14]:


encoder = torch.load(encode_path)
decoder = torch.load(decode_path)
encoder.cuda()
decoder.cuda()
print(encoder)
print(decoder)


# In[15]:


f = open(args.output_path,mode='w')
for i,text in enumerate(tqdm(valid_loader)):
    tmp = text['text'][0]
    batch_num = text['text'].shape[0]
    btach_num = batch_num
    for j in range(1,batch_num):
        b = text['text'][j]
        tmp=torch.cat( (tmp,b) ,dim = 0)
    
    tmp = tmp.view(batch_num,-1) # tmp (batch , seq_len)
    tmp = V(tmp).cuda()
    encode_output,encode_hidden = encoder(tmp)
    # encode_output.shape = [10, 300, 300]
    # encode_hidden.shape = [4, 10, 150]
    
    start_out = torch.zeros([batch_num,1],dtype=torch.long).cuda()
    for m in range(batch_num):
        start_out[m] = text['text'][m][0]
    start_out = torch.unsqueeze(start_out,-1)
    
    for m in range(batch_num):
        e_hidden = encode_hidden[:,m,:]
        e_hidden = torch.unsqueeze(e_hidden,1).contiguous()
        ec = encode_output[m,::]
        ec = torch.unsqueeze(ec,0).contiguous()
        out_words=[]
        #print(start_out[m].shape) = [1, 1]
        #print(hidden.shape) = [4, 1, 150]
        #print(ec.shape) = [1, 300, 300]
        for j in range(80):
            if j==0:
                decoder_output,decoder_hidden = decoder(start_out[m],e_hidden,ec,batch_num)
                #print(e_hidden)
                #print(decoder_hidden)
                #print(decoder_output.shape) [1, 1, 113378]
                #print(decoder_hidden.shape) [4, 1, 150]
                #decoder_output = F.softmax(decoder_output,2)
                max_prob,max_index = torch.max(decoder_output,2)
                output_words = embed.vocab[max_index.item()]
                output_index = max_index.item()
            else:
                input_vocab = torch.tensor([output_index])
                input_vocab = input_vocab.unsqueeze(-1)
                input_vocab = input_vocab.cuda()
                decoder_output,decoder_hidden = decoder(input_vocab ,decoder_hidden,ec,batch_num)
                max_prob,max_index = torch.max(decoder_output,2)
                output_words = embed.vocab[max_index.item()]
                output_index = max_index.item()
            if output_index==2: #<eos>
                break
            if output_index==3 or output_index==1 or output_index==0: #<unk>
                continue
            out_words.append(output_words)
        out_str = " ".join(out_words)
        ans_list={}
        ans_list['id'] = text['id'][m]
        ans_list['predict'] = out_str
        f.write(json.dumps(ans_list) + "\n")
        
    

f.close()


# In[ ]:



# In[ ]:




