#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import json
import os
import numpy as np
import string
import torch
from torch import nn
from torch.autograd import Variable as V
import torch.optim as optim
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from argparse import ArgumentParser
from transformers import BertTokenizer, BertForQuestionAnswering ,BertModel,BertPreTrainedModel
import random
from torch.nn.utils.rnn import pad_sequence


# In[ ]:


train_file='../adl/hw2/data/train.json'
dev_file = '../adl/hw2/data/dev.json'


# In[ ]:


with open(train_file) as f:
    train_data = json.load(f)
with open(dev_file) as f:
    dev_data = json.load(f)


# In[ ]:


tokenizer = BertTokenizer.from_pretrained("bert-base-chinese",do_lower_case=True)


# In[ ]:



train_list=[]
for i in range(len(train_data['data'])):
    para_text = (train_data['data'][i]['paragraphs'])
    for j in range(len(para_text)):
        q_context = train_data['data'][i]['paragraphs'][j]['context']
        q_list = train_data['data'][i]['paragraphs'][j]['qas']
        for k in range(len(q_list)):
            train_dict={}
            train_dict['context'] = q_context
            train_dict['qas'] = q_list[k]
            train_list.append(train_dict)
            
        
dev_list=[]
for i in range(len(dev_data['data'])):
    para_text = (dev_data['data'][i]['paragraphs'])
    for j in range(len(para_text)):
        q_context = dev_data['data'][i]['paragraphs'][j]['context']
        q_list = dev_data['data'][i]['paragraphs'][j]['qas']
        for k in range(len(q_list)):
            train_dict={}
            train_dict['context'] = q_context
            train_dict['qas'] = q_list[k]
            dev_list.append(train_dict)       
    


# In[ ]:


for k in range(len(train_list)):
    train_list[k]['context'] = " ".join(train_list[k]['context'])
    train_list[k]['qas']['question'] = " ".join(train_list[k]['qas']['question'])
for k in range(len(dev_list)):
    dev_list[k]['context'] = " ".join(dev_list[k]['context'])
    dev_list[k]['qas']['question'] = " ".join(dev_list[k]['qas']['question'])


# In[ ]:


print(train_list[0]['context'])
print(train_list[0]['qas'])


# In[ ]:


class BertDataset(Dataset):
    def __init__(self, train_data_set,tokenizer):
        self.t_data = train_data_set
    def __getitem__(self, idx):
        context = self.t_data[idx]['context']
        question = self.t_data[idx]['qas']['question']
        
        
        answer_start = self.t_data[idx]['qas']['answers'][0]['answer_start']+1
        answer_end = answer_start+(len(self.t_data[idx]['qas']['answers'][0]['text'] ) -1)
        if answer_start==0:
            answer_start=-1
            answer_end=-1
        
        tmp_a=""
        tmp_b=""
        tmp_b = tokenizer.tokenize(question) + ["[SEP]"]
        len_b = len(tmp_b)
        if len_b >=50:
            len_b = 51
            tmp_b = tmp_b[0:50]
            tmp_b = tmp_b+["[SEP]"]
        
        word_pieces = ["[CLS]"]
        word_pieces = word_pieces + tokenizer.tokenize(context) + ["[SEP]"]
        len_a = len(word_pieces)
        if len_a >=(512-len_b-1):
            len_a=(512-len_b)
            tmp_a = word_pieces[0:(len_a-1)]
            tmp_a = tmp_a+["[SEP]"]
        else:
            tmp_a = word_pieces
       
        if answer_end >= len_a:
            answer_start=-1
            answer_end=-1
    
        if self.t_data[idx]['qas']['answerable'] == True and answer_start!=-1:
            answer_able_tensor = torch.tensor(1)
        elif self.t_data[idx]['qas']['answerable'] == True and answer_start==-1:
            answer_able_tensor = torch.tensor(-1)
        else:
            answer_able_tensor = torch.tensor(0)
        
        tmp = tmp_a+tmp_b
        ids = tokenizer.convert_tokens_to_ids(tmp)
        tokens_tensor = torch.tensor(ids)
        
        segments_tensor = torch.tensor([0] * len_a + [1] * len_b, dtype=torch.long)
        ans_start_tensor = torch.tensor(answer_start)
        end_start_tensor = torch.tensor(answer_end)
        return (tokens_tensor, segments_tensor,answer_able_tensor,ans_start_tensor,end_start_tensor)
    def __len__(self):
        return len(self.t_data)


# In[ ]:


def create_batch(samples):
    tokens_tensors =[]
    segments_tensors =[]
    for ss in samples:
        tokens_tensors.append(ss[0])
        segments_tensors.append(ss[1])
    
    label_ids = torch.stack([s[2] for s in samples])
    start_ids = torch.stack([s[3] for s in samples])
    end_ids = torch.stack([s[4] for s in samples])
    
    


    tokens_tensors = pad_sequence(tokens_tensors,batch_first=True)
    segments_tensors = pad_sequence(segments_tensors,batch_first=True)
    
    masks_tensors = torch.zeros(tokens_tensors.shape,dtype=torch.long)
    masks_tensors = masks_tensors.masked_fill(tokens_tensors != 0, 1)
    
    return tokens_tensors, segments_tensors, masks_tensors, label_ids,start_ids,end_ids


# In[ ]:


BATCH_SIZE=9
EPOCHS = 10
bert_set = BertDataset(train_list,tokenizer)
bert_valid = BertDataset(dev_list,tokenizer)
train_loader = torch.utils.data.DataLoader(dataset=bert_set, batch_size=BATCH_SIZE, shuffle=True,collate_fn=create_batch)
valid_loader = torch.utils.data.DataLoader(dataset=bert_valid, batch_size=1, shuffle=False,collate_fn=create_batch)


# In[ ]:


class MybertQA(BertPreTrainedModel):
    def __init__(self,config):
        super(MybertQA, self).__init__(config)
        self.bert = BertModel(config)
        self.start_outputs = nn.Linear(config.hidden_size, 1)
        self.answerable_outputs = nn.Linear(config.hidden_size, 2)
        self.end_outputs = nn.Linear(config.hidden_size, 1)
        self.dropout = nn.Dropout(0.1)
        self.init_weights()
    def forward(self,
        input_ids,
        attention_mask,
        token_type_ids,
        start_positions,
        end_positions,
        label_ids):
        
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        
        
        start_logits = self.start_outputs(sequence_output)
        end_logits = self.end_outputs(sequence_output)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)
    
        
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        answer_able = self.answerable_outputs(pooled_output)
        answer_able = torch.squeeze(answer_able,-1)
        
        
        loss_fct = CrossEntropyLoss(ignore_index=-1)
        start_loss = loss_fct(start_logits, start_positions)
        end_loss = loss_fct(end_logits, end_positions)
        start_loss = start_loss 
        end_loss = end_loss
        loss_p = loss_fct(answer_able, label_ids)
        return (start_loss+end_loss+loss_p)/3
        
        #(input_ids.shape) (batch size , seq_len)
        #print(outputs[0].shape) #(batch size , seq_len,768)
        #print(outputs[1].shape) #(batch size , 768)
        
        


model = MybertQA.from_pretrained("bert-base-chinese")
model = model.cuda()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
print(model)


# In[ ]:


model.train()
print("t4")
for epoch in range(EPOCHS):
    running_loss=0
    for step, (b_w,b_x, b_y,b_z,b_s,b_e) in enumerate (tqdm(train_loader)):
        b_w = V(b_w).cuda() # tokens_tensors [batch , seq_len]
        b_x = V(b_x).cuda() # segments_tensors[batch , seq_len]
        b_y = V(b_y).cuda() # masks_tensors [batch , seq_len]
        b_z = V(b_z).cuda() # label_ids [batch]
        b_s = V(b_s).cuda() # [batch]
        b_e = V(b_e).cuda() # [batch]
        loss = model( input_ids=b_w ,attention_mask=b_y,token_type_ids=b_x, start_positions=b_s,end_positions=b_e,label_ids=b_z)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        
        
    print("train loss:",running_loss/len(train_loader))
    
    
    torch.save(model, 'model_14_3L_CN'+str(epoch)+'.pt')
    for step, (b_w,b_x, b_y,b_z,b_s,b_e) in enumerate (tqdm(valid_loader)):
       
        b_w = V(b_w).cuda() # tokens_tensors
        b_x = V(b_x).cuda() # segments_tensors
        b_y = V(b_y).cuda() # masks_tensors
        b_z = V(b_z).cuda() # label_ids
        b_s = V(b_s).cuda()
        b_e = V(b_e).cuda()
        loss = model( input_ids=b_w ,attention_mask=b_y,token_type_ids=b_x, start_positions=b_s,end_positions=b_e,label_ids=b_z)
        optimizer.zero_grad()
        running_loss += loss.item()
    print("valid loss:",running_loss/len(valid_loader))
    
    