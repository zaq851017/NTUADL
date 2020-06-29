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
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


# In[9]:


encode_path='model3_encoder.pkl'
decode_path='model3_decoder.pkl'
#output_path='test_attention_valid_soft.jsonl'

with open('./valid.pkl', 'rb') as file:
    valid =pickle.load(file)
with open('./embedding.pkl', 'rb') as file:
    embed = pickle.load(file)


# In[10]:



BATCH_SIZE = 1
WORD_VECTOR_LENGTH = 300
hidden_size = 150
num_layers=2
vocab_size = embed.vectors.shape[0]
b_dir = 2


# In[11]:


valid_loader = torch.utils.data.DataLoader(dataset=valid, batch_size=BATCH_SIZE, shuffle=False,collate_fn=valid.collate_fn)


# In[ ]:





# In[12]:


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
        return output,hidden,attn_weights


# In[14]:


encoder = torch.load(encode_path)
decoder = torch.load(decode_path)
encoder.cuda()
decoder.cuda()
print(encoder)
print(decoder)


# In[15]:



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
    attn_matrix = torch.zeros(encode_output.shape[1],20)
    for m in range(batch_num):
        e_hidden = encode_hidden[:,m,:]
        e_hidden = torch.unsqueeze(e_hidden,1).contiguous()
        ec = encode_output[m,::]
        ec = torch.unsqueeze(ec,0).contiguous()
        out_words=[]
        #print(start_out[m].shape) = [1, 1]
        #print(hidden.shape) = [4, 1, 150]
        #print(ec.shape) = [1, 300, 300]
        predict_len  = 0
        for j in range(80):
            if j==0:
                decoder_output,decoder_hidden,attn = decoder(start_out[m],e_hidden,ec,batch_num)
                #print(e_hidden)
                #print(decoder_hidden)
                #print(decoder_output.shape) [1, 1, 113378]
                #print(decoder_hidden.shape) [4, 1, 150]
                #decoder_output = F.softmax(decoder_output,2)
                max_prob,max_index = torch.max(decoder_output,2)
                output_words = embed.vocab[max_index.item()]
                output_index = max_index.item()
                #out_words.append(output_words)
            else:
                input_vocab = torch.tensor([output_index])
                input_vocab = input_vocab.unsqueeze(-1)
                input_vocab = input_vocab.cuda()
                decoder_output,decoder_hidden,attn = decoder(input_vocab ,decoder_hidden,ec,batch_num)
                max_prob,max_index = torch.max(decoder_output,2)
                output_words = embed.vocab[max_index.item()]
                output_index = max_index.item()
            if output_index==2: #<eos>
                break
            if output_index==3 or output_index==1 or output_index==0: #<unk>
                continue
            out_words.append(output_words)
            attn_matrix[:,predict_len] = attn
            predict_len = predict_len + 1
        out_str = " ".join(out_words)
        ans_list={}
        ans_list['id'] = text['id'][m]
        ans_list['predict'] = out_str
    if i==0:
        break
   
        
    


# In[17]:


valid_data = []
for i in range(297):
    valid_data.append(embed.vocab[valid[0]['text'][i] ])


# In[18]:


#attn = attn.cpu()
#attn = torch.squeeze(attn,0)
fig = plt.figure(figsize=(20,227))
ax = fig.add_subplot(111)
cax = ax.matshow(attn_matrix.detach().numpy(), cmap='bone')
fig.colorbar(cax)

    # Set up axes
#ax.set_xticklabels(valid_data+[''], rotation=90)
#ax.set_yticklabels(out_words+[''])
ax.set_xticklabels(['']+out_words)
ax.set_yticklabels(['']+valid_data, rotation=90)
    # Show label at every tick
ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

plt.show()


# In[ ]:





# In[ ]:


print(out_words)


# In[ ]:




