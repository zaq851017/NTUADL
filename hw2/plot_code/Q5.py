#!/usr/bin/env python
# coding: utf-8

# In[2]:


import matplotlib.pyplot as plt
import numpy as np
import json


# In[53]:


with open('./data/train.json') as f:
    test_data = json.load(f)
train_list=[]
for i in range(len(test_data['data'])):
    para_text = (test_data['data'][i]['paragraphs'])
    for j in range(len(para_text)):
        q_context = test_data['data'][i]['paragraphs'][j]['context']
        q_list = test_data['data'][i]['paragraphs'][j]['qas']
        for k in range(len(q_list)):
            train_dict={}
            train_dict['context'] = q_context
            train_dict['qas'] = q_list[k]
            train_list.append(train_dict)

for i in range(len(train_list)):
    t_len = len(train_list[i]['qas']['answers'][0]['text'])
    if t_len==0:
        continue
    len_list.append(t_len)


"""
x = [i for i in range(31)]
len_list = [0 for n in range(31)]
for i in train.items():
    if len(i[1])==0:
        continue
    len_list[len(i[1])] = len_list[len(i[1])]+1
sum_l = sum(len_list)
for i in range(len(len_list)):
    len_list[i] = len_list[i]/sum_l
print(sum(len_list))
print(x)
"""


# In[59]:


plt.xlabel("Length")
plt.ylabel("Count(%)")
plt.hist(len_list,bins=15,range=(0,100),normed=True ,cumulative=True,histtype=u'bar',rwidth=0.5)


# In[48]:


print(len_list)


# In[ ]:




