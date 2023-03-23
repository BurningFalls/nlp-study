#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/BurningFalls/nlp-study/blob/main/BERT%20(E-book)/1_BERT_%EB%A7%9B%EB%B3%B4%EA%B8%B0.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# In[1]:


pip install transformers


# In[2]:


import pandas as pd
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased") # bert-base의 토크나이저


# In[3]:


result = tokenizer.tokenize('Here is the sentence I want embeddings for.')
print(result)


# In[4]:


print(tokenizer.vocab['here'])


# In[5]:


# print(tokenizer.vocab['embeddings']) -> KeyError: 'embeddings'


# In[6]:


print(tokenizer.vocab['em'])
print(tokenizer.vocab['##bed'])
print(tokenizer.vocab['##ding'])
print(tokenizer.vocab['##s'])


# In[7]:


# BERT의 단어 집합을 vocabulary.txt에 저장
with open('vocabulary.txt', 'w') as f:
  for token in tokenizer.vocab.keys():
    f.write(token + '\n')


# In[8]:


df = pd.read_fwf('vocabulary.txt', header=None)
df


# In[9]:


print(f'단어 집합의 크기 : {len(df)}')


# In[10]:


df.loc[4667].values[0]


# In[12]:


print(df.loc[0].values[0])
print(df.loc[100].values[0])
print(df.loc[101].values[0])
print(df.loc[102].values[0])
print(df.loc[103].values[0])

