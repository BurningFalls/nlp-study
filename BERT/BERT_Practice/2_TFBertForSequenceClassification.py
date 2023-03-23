#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/BurningFalls/nlp-study/blob/main/BERT%20%EC%8B%A4%EC%8A%B5/BERT_TFBertForSequenceClassification.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# In[1]:


get_ipython().system('pip install transformers')


# In[2]:


import transformers
import pandas as pd
import numpy as np
import re
import urllib.request
import tensorflow as tf
from transformers import BertTokenizerFast, TFBertForSequenceClassification
from keras.callbacks import EarlyStopping


# In[3]:


urllib.request.urlretrieve("https://raw.githubusercontent.com/e9t/nsmc/master/ratings_train.txt", filename="ratings_train.txt")
urllib.request.urlretrieve("https://raw.githubusercontent.com/e9t/nsmc/master/ratings_test.txt", filename="ratings_test.txt")


# In[4]:


train_data = pd.read_table('ratings_train.txt')
test_data = pd.read_table('ratings_test.txt')


# In[5]:


# delete redundant data
train_data.drop_duplicates(subset=['document'], inplace=True)
# delete null values
train_data = train_data.dropna(how = 'any')
test_data = test_data.dropna(how = 'any')


# In[6]:


tokenizer = BertTokenizerFast.from_pretrained("klue/bert-base")


# In[7]:


train_X_list = train_data['document'].tolist()
train_X = tokenizer(train_X_list, truncation=True, padding=True)
train_y = train_data['label'].tolist()

test_X_list = test_data['document'].tolist()
test_X = tokenizer(test_X_list, truncation=True, padding=True)
test_y = test_data['label'].tolist()


# In[8]:


train_dataset = tf.data.Dataset.from_tensor_slices((
    dict(train_X),
    train_y
))

val_dataset = tf.data.Dataset.from_tensor_slices((
    dict(test_X),
    test_y
))


# In[9]:


optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5)
model = TFBertForSequenceClassification.from_pretrained("klue/bert-base", num_labels=2, from_pt=True)
model.compile(optimizer=optimizer, loss=model.hf_compute_loss, metrics=['accuracy'])


# In[10]:


callback_earlystop = EarlyStopping(
    monitor="val_accuracy", 
    min_delta=0.001,
    patience=2)

model.fit(
    train_dataset.shuffle(10000).batch(32), epochs=2, batch_size=64,
    validation_data = val_dataset.shuffle(10000).batch(64),
    callbacks = [callback_earlystop]
)


# In[ ]:


model.evaluate(val_dataset.batch(1024))


# In[ ]:


model.save_pretrained('nsmc_model/bert-base')
tokenizer.save_pretrained('nsmc_model/bert-base')


# In[ ]:


from transformers import TextClassificationPipeline

# 로드하기
loaded_tokenizer = BertTokenizerFast.from_pretrained('nsmc_model/bert-base')
loaded_model = TFBertForSequenceClassification.from_pretrained('nsmc_model/bert-base')

text_classifier = TextClassificationPipeline(
    tokenizer=loaded_tokenizer, 
    model=loaded_model, 
    framework='tf',
    return_all_scores=True
)


# In[ ]:


text_classifier('뭐야 이 평점들은.... 나쁘진 않지만 10점 짜리는 더더욱 아니잖아')[0]


# In[ ]:


text_classifier('오랜만에 평점 로긴했네ㅋㅋ 킹왕짱 쌈뽕한 영화를 만났습니다 강렬하게 육쾌함')[0]

