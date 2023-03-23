#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/BurningFalls/nlp-study/blob/main/BERT%20(E-book)/4_%EA%B5%AC%EA%B8%80_BERT%EC%9D%98_%EB%8B%A4%EC%9D%8C_%EB%AC%B8%EC%9E%A5_%EC%98%88%EC%B8%A1(NSP_Next_Sentence_Prediction).ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# In[1]:


pip install transformers


# In[2]:


import tensorflow as tf
from transformers import TFBertForNextSentencePrediction
from transformers import AutoTokenizer


# In[3]:


model = TFBertForNextSentencePrediction.from_pretrained('bert-base-uncased')
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')


# In[4]:


prompt = "In Italy, pizza served in formal settings, such as at a restaurant, is presented unsliced."
next_sentence = "pizza is eaten with the use of a knife and fork. In casual settings, however, it is cut into wedges to be eaten while held in the hand."


# In[5]:


encoding = tokenizer(prompt, next_sentence, return_tensors='tf')


# In[6]:


print(encoding['input_ids']) # 정수 인코딩 결과


# In[7]:


print(f'{tokenizer.cls_token} : {tokenizer.cls_token_id}')
print(f'{tokenizer.sep_token} : {tokenizer.sep_token_id}')


# In[8]:


print(tokenizer.decode(encoding['input_ids'][0]))


# In[9]:


print(encoding['token_type_ids']) # 세그먼트 인코딩 결과


# In[10]:


logits = model(encoding['input_ids'], token_type_ids=encoding['token_type_ids'])[0]
softmax = tf.keras.layers.Softmax()
probs = softmax(logits)
print(probs)


# In[11]:


# 0: 실질적으로 이어지는 두 개의 문장의 레이블
# 1: 실질적으로 이어지지 않는 두 개의 문장의 레이블
print(f'최종 예측 레이블 : {tf.math.argmax(probs, axis=-1).numpy()}')


# In[12]:


# 상관없는 두 개의 문장
prompt = "In Italy, pizza served in formal settings, such as at a restaurant, is presented unsliced."
next_sentence = "The sky is blue due to the shorter wavelength of blue light."
encoding = tokenizer(prompt, next_sentence, return_tensors='tf')

logits = model(encoding['input_ids'], token_type_ids=encoding['token_type_ids'])[0]

softmax = tf.keras.layers.Softmax()
probs = softmax(logits)
print('최종 예측 레이블 :', tf.math.argmax(probs, axis=-1).numpy())

