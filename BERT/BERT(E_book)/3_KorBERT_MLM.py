#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/BurningFalls/nlp-study/blob/main/BERT%20(E-book)/3_%ED%95%9C%EA%B5%AD%EC%96%B4_BERT%EC%9D%98_%EB%A7%88%EC%8A%A4%ED%81%AC%EB%93%9C_%EC%96%B8%EC%96%B4_%EB%AA%A8%EB%8D%B8(MLM_Masked_Language_Model)_%EC%8B%A4%EC%8A%B5.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# In[1]:


pip install transformers


# In[3]:


from transformers import TFBertForMaskedLM
from transformers import AutoTokenizer


# In[4]:


model = TFBertForMaskedLM.from_pretrained('klue/bert-base', from_pt=True)
tokenizer = AutoTokenizer.from_pretrained('klue/bert-base')


# In[5]:


inputs = tokenizer('축구는 정말 재미있는 [MASK]다.', return_tensors='tf')


# In[6]:


print(inputs['input_ids']) # 정수 인코딩 결과


# In[7]:


print(inputs['token_type_ids']) # 세그먼트 인코딩 결과


# In[8]:


print(inputs['attention_mask']) # 어텐션 마스크


# In[9]:


from transformers import FillMaskPipeline
pip = FillMaskPipeline(model=model, tokenizer=tokenizer)


# In[10]:


pip('축구는 정말 재미있는 [MASK]다.')


# In[11]:


pip('어벤져스는 정말 재미있는 [MASK]다.')


# In[12]:


pip('나는 오늘 아침에 [MASK]에 출근을 했다.')

