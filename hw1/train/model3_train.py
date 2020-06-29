#!/usr/bin/env python
# coding: utf-8

# In[20]:


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


# In[21]:



with open('../datasets/seq2seq/valid.pkl', 'rb') as file:
    valid =pickle.load(file)
with open('../datasets/seq2seq/embedding.pkl', 'rb') as file:
    embed = pickle.load(file)


# In[22]:


BATCH_SIZE = 17
WORD_VECTOR_LENGTH = 300
EPOCH = 20
hidden_size = 150
num_layers=2
vocab_size = embed.vectors.shape[0]
b_dir = 2


# In[23]:


valid_loader = torch.utils.data.DataLoader(dataset=valid, batch_size=BATCH_SIZE, shuffle=False,collate_fn=valid.collate_fn)


# In[24]:


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

encoder = ENCODER()
encoder.cuda()
print(encoder)


# In[25]:


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
        tmp_hidden = tmp_hidden.transpose(0, 1) # hidden = [9, 4, 150]
        tmp_hidden = tmp_hidden.contiguous().view(batch_num,1,600) # hidden = [9, 1, 600]
        tt = torch.cat( (embed_vector,tmp_hidden),2 )
        tt = self.attn(tt)
        attn_weights = F.softmax(tt, dim=2) #attn_weights = ([9, 1, 300])
        attn_weights = attn_weights.narrow(2, 0, encoder_output.shape[1]) #attn_weights = ([9, 1, 299])
        attn_applied = torch.bmm(attn_weights , encode_output )  # attn_applied =[9, 1, 300]
        attn_embed = torch.cat((attn_applied,embed_vector),2 )
        output,hidden = self.rnn(attn_embed,hidden)
        output = self.fc(output)
        return output,hidden
        #output = torch.Size([9, 1, 113378])
        #hidden = torch.Size([4, 9, 150])
    
att_decoder = AttnDecoderRNN()
att_decoder.cuda()
print(att_decoder)


# In[26]:


LR = 0.001 
opt1 = torch.optim.Adam(encoder.parameters(), lr=LR)
opt2 = torch.optim.Adam(att_decoder.parameters(), lr=LR)
loss_func = torch.nn.CrossEntropyLoss() 


# In[30]:


for epoch in range(EPOCH):
    train_loss = 0.0
    valid_loss = 0.0
    for i,text in enumerate(tqdm(valid_loader)):
        opt1.zero_grad()
        opt2.zero_grad()
        tmp = text['text'][0]
        tmp2 = text['summary'][0]
        btach_num = text['text'].shape[0]
        for j in range(1,btach_num):
            b = text['text'][j]
            c = text['summary'][j]
            tmp=torch.cat( (tmp,b) ,dim = 0)
            tmp2=torch.cat( (tmp2,c) ,dim = 0)
        tmp = tmp.view(btach_num,-1)
        tmp2 = tmp2.view(btach_num,-1)
        tmp = V(tmp).cuda()
        tmp2 = V(tmp2).cuda()
        #tmp = V(tmp)
        #tmp2 = V(tmp2)
        target_len = tmp2.shape[1]
        encode_output,encode_hidden = encoder(tmp)
        hidden = encode_hidden
        text_len = encode_output.shape[1]
        # encode_output.shape = [9, 299, 300] = [batch , input_len , hidden*dir]
        # encode_hidden = [4, 9, 150] = [num_layer , batch,hidden ]
        # tmp2 = [batch , summary_max_len ]
        ans = text['summary'][:,1:target_len]
        zero_ten = torch.zeros(btach_num,1,dtype=torch.long)
        ans = torch.cat((ans,zero_ten),dim=1)
        ans = ans.cuda()
        #outputs = torch.zeros(btach_num , target_len, vocab_size)
        outputs = torch.zeros(btach_num , target_len, vocab_size).cuda()
        for j in range(target_len):
            ls = tmp2[:,j].unsqueeze(-1)
            decode_output,hidden=att_decoder(ls,hidden,encode_output,btach_num)
            decode_output = torch.squeeze(decode_output,1)
            outputs[:,j,:] = decode_output

        outputs = outputs.view((-1,vocab_size))
        ans = ans.view(-1)
        loss = loss_func(outputs,ans)
        train_loss += loss.item()
        loss.backward() 
        opt1.step()
        opt2.step()
    torch.save(encoder, 'attn_encoder_b17'+str(epoch)+'.pkl')
    torch.save(att_decoder, 'attn_decoder_b17'+str(epoch)+'.pkl')
    for i,text in enumerate(tqdm(valid_loader)):
        opt1.zero_grad()
        opt2.zero_grad()
        tmp = text['text'][0]
        tmp2 = text['summary'][0]
        btach_num = text['text'].shape[0]
        for j in range(1,btach_num):
            b = text['text'][j]
            c = text['summary'][j]
            tmp=torch.cat( (tmp,b) ,dim = 0)
            tmp2=torch.cat( (tmp2,c) ,dim = 0)
        tmp = tmp.view(btach_num,-1)
        tmp2 = tmp2.view(btach_num,-1)
        tmp = V(tmp).cuda()
        tmp2 = V(tmp2).cuda()
        #tmp = V(tmp)
        #tmp2 = V(tmp2)
        target_len = tmp2.shape[1]
        encode_output,encode_hidden = encoder(tmp)
        hidden = encode_hidden
        text_len = encode_output.shape[1]
        # encode_output.shape = [9, 299, 300] = [batch , input_len , hidden*dir]
        # encode_hidden = [4, 9, 150] = [num_layer , batch,hidden ]
        # tmp2 = [batch , summary_max_len ]
        ans = text['summary'][:,1:target_len]
        zero_ten = torch.zeros(btach_num,1,dtype=torch.long)
        ans = torch.cat((ans,zero_ten),dim=1)
        ans = ans.cuda()
        #outputs = torch.zeros(btach_num , target_len, vocab_size)
        outputs = torch.zeros(btach_num , target_len, vocab_size).cuda()
        for j in range(target_len):
            ls = tmp2[:,j].unsqueeze(-1)
            decode_output,hidden=att_decoder(ls,hidden,encode_output,btach_num)
            decode_output = torch.squeeze(decode_output,1)
            outputs[:,j,:] = decode_output

        outputs = outputs.view((-1,vocab_size))
        ans = ans.view(-1)
        loss = loss_func(outputs,ans)
        valid_loss += loss.item()
    print(train_loss/len(train_loader))
    print(valid_loss/len(valid_loader))
    


# In[ ]:




