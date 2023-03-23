#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/BurningFalls/nlp-study/blob/main/BERT%20(E-book)/2_%EA%B5%AC%EA%B8%80_BERT%EC%9D%98_%EB%A7%88%EC%8A%A4%ED%81%AC%EB%93%9C_%EC%96%B8%EC%96%B4_%EB%AA%A8%EB%8D%B8(MLM_Masked_Language_Model)_%EC%8B%A4%EC%8A%B5.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# In[1]:


pip install transformers


# In[2]:


from transformers import TFBertForMaskedLM
from transformers import AutoTokenizer


# In[3]:


model = TFBertForMaskedLM.from_pretrained('bert-large-uncased')
tokenizer = AutoTokenizer.from_pretrained('bert-large-uncased')


# In[4]:


inputs = tokenizer('Soccer is a really fun [MASK].', return_tensors='tf')


# In[5]:


print(inputs['input_ids']) # 정수 인코딩 결과


# In[6]:


print(inputs['token_type_ids']) # 세그먼트 인코딩 결과


# In[7]:


print(inputs['attention_mask']) # 어텐션 마스크


# In[8]:


from transformers import FillMaskPipeline
pip = FillMaskPipeline(model=model, tokenizer=tokenizer)


# In[9]:


pip('Soccer is a really fun [MASK]')


# In[10]:


pip('The Avengers is a really fun [MASK].')


# In[11]:


pip('I went to [MASK] this morning.')

