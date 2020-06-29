#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import numpy as np
import json


# In[3]:


with open('result01.json') as f:
    dev_01 = json.load(f)
with open('result03.json') as f:
    dev_03 = json.load(f)
with open('result05.json') as f:
    dev_05 = json.load(f)
with open('result07.json') as f:
    dev_07 = json.load(f)
with open('result09.json') as f:
    dev_09 = json.load(f)


# In[13]:


x = [0.1,0.3,0.5,0.7,0.9]
F1_u=[]
F1_a=[]
F1_o=[]
EM_u=[]
EM_a=[]
EM_o=[]
F1_o.extend([dev_01['overall']['f1'],dev_03['overall']['f1'],dev_05['overall']['f1'],dev_07['overall']['f1'],dev_09['overall']['f1']])
F1_u.extend([dev_01['unanswerable']['f1'],dev_03['unanswerable']['f1'],dev_05['unanswerable']['f1'],dev_07['unanswerable']['f1'],dev_09['unanswerable']['f1']])
F1_a.extend([dev_01['answerable']['f1'],dev_03['answerable']['f1'],dev_05['answerable']['f1'],dev_07['answerable']['f1'],dev_09['answerable']['f1']])
EM_o.extend([dev_01['overall']['em'],dev_03['overall']['em'],dev_05['overall']['em'],dev_07['overall']['em'],dev_09['overall']['em']])
EM_u.extend([dev_01['unanswerable']['em'],dev_03['unanswerable']['em'],dev_05['unanswerable']['em'],dev_07['unanswerable']['em'],dev_09['unanswerable']['em']])
EM_a.extend([dev_01['answerable']['em'],dev_03['answerable']['em'],dev_05['answerable']['em'],dev_07['answerable']['em'],dev_09['answerable']['em']])


# In[21]:


plt.figure(figsize=(15,10),dpi=100,linewidth = 2)
plt.plot(x,F1_o,'o-',color = 'r', label="overall")
plt.plot(x,F1_u,'o-',color = 'g', label="unanswerable")
plt.plot(x,F1_a,'o-',color = 'b', label="answerable")
plt.legend(loc='upper right')
plt.title("F1")
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.show()


# In[22]:


plt.figure(figsize=(15,10),dpi=100,linewidth = 2)
plt.plot(x,EM_o,'o-',color = 'r', label="overall")
plt.plot(x,EM_u,'o-',color = 'g', label="unanswerable")
plt.plot(x,EM_a,'o-',color = 'b', label="answerable")
plt.legend(loc='upper right')
plt.title("EM")
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.show()


# In[ ]:




