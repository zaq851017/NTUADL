#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import json
import nltk
import os
import numpy as np
import string
import torch
from torch import nn
from torch.autograd import Variable as V
import torch.optim as optim
import torch.nn.functional as F 
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from argparse import ArgumentParser
#nltk.download()

glove_path ='./glove.6B.300d.txt'
model_path='./model1.pkl'
# In[ ]:
WORD_VECTOR_LENGTH = 300
hidden_size = 50
class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()

        self.rnn = nn.LSTM(         # if use nn.RNN(), it hardly learns
            input_size=WORD_VECTOR_LENGTH,
            hidden_size=50,         # rnn hidden unit
            num_layers=3,           # number of rnn layer
            batch_first=True,       # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
            bidirectional=True
        )

        self.out = nn.Linear(hidden_size*2, 1)

    def forward(self, x):
        # x shape (batch, time_step, input_size)
        # r_out shape (batch, time_step, output_size)
        # h_n shape (n_layers, batch, hidden_size)
        # h_c shape (n_layers, batch, hidden_size)
        r_out, (h_n, h_c) = self.rnn(x, None)   # None represents zero initial hidden state

        # choose r_out at the last time step
        out = self.out(r_out)
        #return F.sigmoid(out)
        return out

# In[ ]:


rnn = torch.load(model_path)
rnn.cuda()
print(rnn)


# In[ ]:

parser = ArgumentParser()
parser.add_argument('--input_data_path')
parser.add_argument('--output_path')
args  = parser.parse_args()

test_data = []
with open(args.input_data_path) as f:
    for lines in f:
        tmp_data = json.loads(lines)
        test_data.append(tmp_data)


# In[ ]:


embeddings_index = {}
f = open(glove_path, encoding='utf8')
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()


# In[ ]:


input_test_data = []
data_length = 175
test_data_bound=[]

for i,text in enumerate(test_data):
    tmp_data=[]
    tmp2_data=[]
    word_len = 0
    sent_tmp = 0
    for sent in range(len(text['sent_bounds'])):
        start_index,end_index = text['sent_bounds'][sent]
        word_list = nltk.word_tokenize(text['text'][start_index:(end_index-1)])
        word_list=[word.lower() for word in word_list]
        for words in word_list:
            if words in embeddings_index:
                tmp_data.append(embeddings_index[words])
            else:
                tmp_data.append(np.zeros(WORD_VECTOR_LENGTH,dtype='float32'))
        for j in range(len(word_list)):
            tmp2_data.append(sent_tmp)
        word_len = word_len + len(word_list)
        sent_tmp = sent_tmp +1
    if(word_len < data_length ): #太短需要補維度
        dist = data_length - word_len
        for j in range(dist):
            tmp2_data.append(-1)
            tmp_data.append(np.zeros(WORD_VECTOR_LENGTH,dtype='float32'))
        
            
        
    elif(word_len > data_length):#太長需要把後面字捨棄
        dist = word_len - data_length
        for j in range(dist):
            tmp_data.pop()
            tmp2_data.pop()
    input_test_data.append(tmp_data)
    test_data_bound.append(tmp2_data)
        


# In[ ]:


print(np.array(input_test_data).shape)
#print(np.array(test_data_bound).shape)
#print(test_data_bound[19999])
test_data_tensor=torch.Tensor(input_test_data)


# In[ ]:


class DL_Dataset(Dataset):#需要继承data.Dataset
    def __init__(self,test):
        # TODO
        # 1. Initialize file path or list of file names.
        self.test = test
    def __getitem__(self, index):
        # TODO
        # 1. Read one data from file (e.g. using numpy.fromfile, PIL.Image.open).
        # 2. Preprocess the data (e.g. torchvision.Transform).
        # 3. Return a data pair (e.g. image and label).
        #这里需要注意的是，第一步：read one data，是一个data
        a = self.test[index]
        return a
        
    def __len__(self):
        # You should change 0 to the total size of your dataset.
        return len(self.test)


# In[ ]:


mydata_set = DL_Dataset(test_data_tensor)
print(mydata_set)
test_loader = torch.utils.data.DataLoader(dataset=mydata_set, batch_size=1, shuffle=False)


# In[ ]:


f = open(args.output_path,mode='w')
for step,b_x in enumerate (tqdm(test_loader)):
    ans_list ={}
    sent_num = max(test_data_bound[step])+1
    
    if sent_num==0:
        ans_list['id'] = test_data[step]['id']
        ans_list['predict_sentence_index'] = [0]
        f.write(json.dumps(ans_list) + "\n")
        continue
    tmp_list = np.zeros(sent_num)
    word_list = np.zeros(sent_num)
    b_x = V(b_x).cuda()
    #print(b_x.shape)
    out = rnn(b_x)
    out = torch.sigmoid(out)
    out = out.cpu()
    out_np = out.data.numpy()
    for i in range(data_length):
        sent = test_data_bound[step][i]
        if sent==-1:
            continue
        tmp_list[sent] = tmp_list[sent] + out_np[0][i]
        word_list[sent] = word_list[sent] +1
    
    for i in range(sent_num):
        tmp_list[i] = tmp_list[i] / word_list[i]
    #print(tmp_list)
    max_sent = (np.where(tmp_list==max(tmp_list))[0])[0]
    #print(max_sent)
    ans_list['id'] = test_data[step]['id']
    ans_list['predict_sentence_index'] = (np.where(tmp_list==max(tmp_list))[0]).tolist()
    f.write(json.dumps(ans_list) + "\n")
        

f.close()


# In[ ]:




