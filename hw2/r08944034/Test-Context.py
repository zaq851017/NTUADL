#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import json
import copy
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
import sys



# In[ ]:


#test_file='./data/dev.json'
model_path='./model_14_3L_CN6.pt'
#output_path ='./dev_m14_6.json'


# In[ ]:


with open(sys.argv[1]) as f:
    test_data = json.load(f)


# In[ ]:


tokenizer = BertTokenizer.from_pretrained("bert-base-chinese",do_lower_case=True)


# In[ ]:


test_list=[]
for i in range(len(test_data['data'])):
    para_text = (test_data['data'][i]['paragraphs'])
    for j in range(len(para_text)):
        q_context = test_data['data'][i]['paragraphs'][j]['context']
        q_list = test_data['data'][i]['paragraphs'][j]['qas']
        for k in range(len(q_list)):
            train_dict={}
            train_dict['context'] = q_context
            train_dict['qas'] = q_list[k]
            test_list.append(train_dict)
output_list  = copy.deepcopy(test_list)


# In[ ]:


for k in range(len(test_list)):
    test_list[k]['context'] = " ".join(test_list[k]['context'])
    test_list[k]['qas']['question'] = " ".join(test_list[k]['qas']['question'])


# In[ ]:


print((test_list[0]['context']))
print((test_list[0]['qas']))


# In[ ]:


class BertDataset(Dataset):
    def __init__(self, train_data_set):
        self.t_data = train_data_set
    def __getitem__(self, idx):
        context = self.t_data[idx]['context']
        question = self.t_data[idx]['qas']['question']
        
        
        
        
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
       
        
        
        
        tmp = tmp_a+tmp_b
        ids = tokenizer.convert_tokens_to_ids(tmp)
        tokens_tensor = torch.tensor(ids)
        
        segments_tensor = torch.tensor([0] * len_a + [1] * len_b, dtype=torch.long)
        
        return (tokens_tensor, segments_tensor)
    def __len__(self):
        return len(self.t_data)


# In[ ]:


def create_batch(samples):
    
    tokens_tensors =[]
    segments_tensors =[]
    for ss in samples:
        tokens_tensors.append(ss[0])
        segments_tensors.append(ss[1])
    tokens_tensors = pad_sequence(tokens_tensors,batch_first=True)
    segments_tensors = pad_sequence(segments_tensors,batch_first=True)
    masks_tensors = torch.zeros(tokens_tensors.shape,dtype=torch.long)
    masks_tensors = masks_tensors.masked_fill(tokens_tensors != 0, 1)
    return tokens_tensors, segments_tensors, masks_tensors


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
        ):
        
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
        
        return answer_able,start_logits,end_logits
        

        
        

        
model = torch.load(model_path)
model = model.cuda()
print(model)

# In[ ]:


BATCH_SIZE=1
bert_set = BertDataset(test_list)
test_loader = torch.utils.data.DataLoader(dataset=bert_set, batch_size=BATCH_SIZE, shuffle=False,collate_fn=create_batch)


# In[ ]:

ans_list ={}
model.eval()
for step, (b_w,b_x, b_y) in enumerate (tqdm(test_loader)):
    b_w = V(b_w).cuda() # tokens_tensors
    b_x = V(b_x).cuda() # segments_tensors
    b_y = V(b_y).cuda() # masks_tensors
    answer_able,start_logits, end_logits = model( input_ids=b_w ,attention_mask=b_y,token_type_ids=b_x)
    #(answer_able.shape) = [1,2]
    #(start_logits.shape) =[1,seq]
    #(end_logits.shape) = [1,seq]
    _1,ans_index = answer_able.topk(1)
    _2,start_index = start_logits.topk(2)
    _3,end_index = end_logits.topk(2)
    ans_able = ans_index[0][0].item()
    start_pos = start_index[0][0].item()
    end_pos = end_index[0][0].item()

    if (end_pos - start_pos) >=15 or start_pos > end_pos:
        start_pos=start_index[0][1].item()
        if (end_pos - start_pos) >=15:
            start_pos=start_index[0][0].item()
            end_pos = start_index[0][1].item()
            
    if end_pos - start_pos > 30:
        ans_able=0
    if ans_able==0:
        ans_list[test_list[step]['qas']['id']]=""
    else:
        ans_list[test_list[step]['qas']['id']] = output_list[step]['context'][(start_pos-1):end_pos]
    """
    print(start_index,end_index)
    print(output_list[step]['context'][(start_index[0][0].item()-1):end_index[0][0].item()])
    print(output_list[step]['context'][(start_index[0][1].item()-1):end_index[0][1].item()])
    print(ans_list[test_list[step]['qas']['id']])
    """
    
f = open(sys.argv[2],mode='w')
json.dump(ans_list,f)
f.close()


# In[ ]:




