#!/usr/bin/env python
# coding: utf-8

# In[2]:


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
#nltk.download()

WORD_VECTOR_LENGTH = 300
hidden_size = 50
# In[3]:


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


rnn = RNN()
rnn.cuda()
print(rnn)


# In[4]:


train_data=[]
tmp_data={}
embeddings_index = {}

with open('./train.jsonl') as f:
    for lines in f:
        tmp_data = json.loads(lines)
        train_data.append(tmp_data)

f = open('../public/hsc/glove.6B.300d.txt', encoding='utf8')
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()


# In[5]:


input_data = []
predict_data= []
data_length = 175

#print(len(train_data[0]['sent_bounds']))
for i,text in enumerate(train_data): #第幾筆的train_data
    para_index = text['extractive_summary']
    haha_data= []
    tmp2_data= []
    word_len = 0
    for sent in range(len(text['sent_bounds'])):#每筆train_data有幾句
        start_index,end_index = text['sent_bounds'][sent]
        if sent == para_index:
            word_list = nltk.word_tokenize(text['text'][start_index:(end_index-1)])
            word_list=[word.lower() for word in word_list]
            for words in word_list:
                if words in embeddings_index:
                    haha_data.append(embeddings_index[words])
                else:
                    haha_data.append(np.zeros(WORD_VECTOR_LENGTH,dtype='float32'))
            for j in range(len(word_list)):
                tmp2_data.append(1)
        else:
            word_list = nltk.word_tokenize(text['text'][start_index:(end_index-1)])
            word_list=[word.lower() for word in word_list]
            for words in word_list:     
                if words in embeddings_index:
                    haha_data.append(embeddings_index[words])
                else:
                    haha_data.append(np.zeros(WORD_VECTOR_LENGTH,dtype='float32'))
            for j in range(len(word_list)):
                tmp2_data.append(0)
        word_len = word_len + len(word_list)
    
    if(word_len < data_length ): #太短需要補維度
        dist = data_length - word_len
        for j in range(dist):
            tmp2_data.append(0)
            haha_data.append(np.zeros(WORD_VECTOR_LENGTH,dtype='float32'))
        
            
        
    elif(word_len > data_length):#太長需要把後面字捨棄
        dist = word_len - data_length
        for j in range(dist):
            haha_data.pop()
            tmp2_data.pop()
    
    input_data.append(haha_data)
    predict_data.append(tmp2_data)

print("input to w2v OK!")
# In[6]:


#input_data_tensor=torch.Tensor(input_data).cuda()
#predict_data_tensor=torch.Tensor(predict_data).cuda()
input_data_tensor=torch.Tensor(input_data)
input_data.clear()
print("input tensor OK!")
predict_data_tensor=torch.Tensor(predict_data)
predict_data.clear()
print("predict tensor OK!")


# In[7]:


class DL_Dataset(Dataset):#需要继承data.Dataset
    def __init__(self,train,label):
        # TODO
        # 1. Initialize file path or list of file names.
        self.train = train
        self.label = label
    def __getitem__(self, index):
        # TODO
        # 1. Read one data from file (e.g. using numpy.fromfile, PIL.Image.open).
        # 2. Preprocess the data (e.g. torchvision.Transform).
        # 3. Return a data pair (e.g. image and label).
        #这里需要注意的是，第一步：read one data，是一个data
        a = self.train[index]
        b = self.label[index]
        return a,b
        
    def __len__(self):
        # You should change 0 to the total size of your dataset.
        return len(self.train)


# In[8]:


BATCH_SIZE  = 10
EPOCH = 5
LR = 0.001 
optimizer = torch.optim.Adam(rnn.parameters(), lr=LR)   # optimize all cnn parameters
loss_func = nn.BCEWithLogitsLoss()                       # the target label is not one-hotted


# In[9]:


mydata_set = DL_Dataset(input_data_tensor,predict_data_tensor)
print(mydata_set)
train_loader = torch.utils.data.DataLoader(dataset=mydata_set, batch_size=BATCH_SIZE, shuffle=True)


# In[23]:


for epoch in range(EPOCH):
    tmp =0
    num = 0
    for step, (b_x, b_y) in enumerate (tqdm(train_loader)):
        b_x = V(b_x).cuda()
        b_y = V(b_y).cuda()
        out = rnn(b_x)
        out = out.squeeze(2)
        #print(b_y.shape)
        #print(out.shape)
        loss = loss_func(out, b_y)
        tmp = tmp+loss
        num = num +1
        optimizer.zero_grad()
        loss.backward() 
        optimizer.step()
    torch.save(rnn, 'rnn_5_word300_bidir_'+str(epoch)+'.pkl')
    print(tmp/num)


# In[ ]:

  


# In[ ]:




