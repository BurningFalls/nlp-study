#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/BurningFalls/nlp-study/blob/main/BERT%20%EC%8B%A4%EC%8A%B5/BERT_KoBERT%EB%A5%BC_%EC%9D%B4%EC%9A%A9%ED%95%9C_KorNLI_%ED%92%80%EC%96%B4%EB%B3%B4%EA%B8%B0_(%EB%8B%A4%EC%A4%91_%ED%81%B4%EB%9E%98%EC%8A%A4_%EB%B6%84%EB%A5%98).ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# In[1]:


pip install transformers


# In[2]:


import transformers
import pandas as pd
import os
import urllib.request
import tensorflow as tf
import numpy as np
from tqdm import tqdm
from sklearn import preprocessing
from transformers import BertTokenizer, TFBertModel
from keras.callbacks import EarlyStopping, ModelCheckpoint


# In[3]:


# 훈련 데이터 다운로드
urllib.request.urlretrieve("https://raw.githubusercontent.com/kakaobrain/KorNLUDatasets/master/KorNLI/multinli.train.ko.tsv", filename="multinli.train.ko.tsv")
urllib.request.urlretrieve("https://raw.githubusercontent.com/kakaobrain/KorNLUDatasets/master/KorNLI/snli_1.0_train.ko.tsv", filename="snli_1.0_train.ko.tsv")


# In[4]:


# 검증 데이터 다운로드
urllib.request.urlretrieve("https://raw.githubusercontent.com/kakaobrain/KorNLUDatasets/master/KorNLI/xnli.dev.ko.tsv", filename="xnli.dev.ko.tsv")


# In[5]:


# 테스트 데이터 다운로드
urllib.request.urlretrieve("https://raw.githubusercontent.com/kakaobrain/KorNLUDatasets/master/KorNLI/xnli.test.ko.tsv", filename="xnli.test.ko.tsv")


# In[6]:


train_snli = pd.read_csv("snli_1.0_train.ko.tsv", sep='\t', quoting=3)
train_xnli = pd.read_csv("multinli.train.ko.tsv", sep='\t', quoting=3)
val_data = pd.read_csv("xnli.dev.ko.tsv", sep='\t', quoting=3)
test_data = pd.read_csv("xnli.test.ko.tsv", sep='\t', quoting=3)


# In[7]:


train_snli.head()


# In[8]:


train_xnli.head()


# In[9]:


# 결합 후 섞기
train_data = train_snli.append(train_xnli)
train_data = train_data.sample(frac=1)


# In[13]:


def drop_na_and_duplciates(df):
  df = df.dropna()
  df = df.drop_duplicates()
  df = df.reset_index(drop=True)
  return df


# In[14]:


# 결측값 및 중복 샘플 제거
train_data = drop_na_and_duplciates(train_data)
val_data = drop_na_and_duplciates(val_data)
test_data = drop_na_and_duplciates(test_data)


# In[15]:


max_seq_len = 128
tokenizer = BertTokenizer.from_pretrained("klue/bert-base")


# In[16]:


def convert_examples_to_features(sent_list1, sent_list2, max_seq_len, tokenizer):

    input_ids, attention_masks, token_type_ids = [], [], []

    for sent1, sent2 in tqdm(zip(sent_list1, sent_list2), total=len(sent_list1)):
        encoding_result = tokenizer.encode_plus(sent1, sent2, max_length=max_seq_len, pad_to_max_length=True)

        input_ids.append(encoding_result['input_ids'])
        attention_masks.append(encoding_result['attention_mask'])
        token_type_ids.append(encoding_result['token_type_ids'])

    input_ids = np.array(input_ids, dtype=int)
    attention_masks = np.array(attention_masks, dtype=int)
    token_type_ids = np.array(token_type_ids, dtype=int)

    return (input_ids, attention_masks, token_type_ids)


# In[17]:


X_train = convert_examples_to_features(train_data['sentence1'], train_data['sentence2'], max_seq_len=max_seq_len, tokenizer=tokenizer)


# In[18]:


X_val = convert_examples_to_features(val_data['sentence1'], val_data['sentence2'], max_seq_len=max_seq_len, tokenizer=tokenizer)


# In[19]:


X_test = convert_examples_to_features(test_data['sentence1'], test_data['sentence2'], max_seq_len=max_seq_len, tokenizer=tokenizer)


# In[20]:


train_label = train_data['gold_label'].tolist()
val_label = val_data['gold_label'].tolist()
test_label = test_data['gold_label'].tolist()


# In[21]:


idx_encode = preprocessing.LabelEncoder()
idx_encode.fit(train_label)

y_train = idx_encode.transform(train_label) # 주어진 고유한 정수로 변환
y_val = idx_encode.transform(val_label) # 고유한 정수로 변환
y_test = idx_encode.transform(test_label) # 고유한 정수로 변환

label_idx = dict(zip(list(idx_encode.classes_), idx_encode.transform(list(idx_encode.classes_))))
idx_label = {value: key for key, value in label_idx.items()}
print(label_idx)
print(idx_label)


# In[22]:


from transformers import TFBertForSequenceClassification


# In[23]:


# TPU 작동을 위한 코드
resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu='grpc://' + os.environ['COLAB_TPU_ADDR'])
tf.config.experimental_connect_to_cluster(resolver)
tf.tpu.experimental.initialize_tpu_system(resolver)


# In[24]:


strategy = tf.distribute.experimental.TPUStrategy(resolver)


# In[25]:


with strategy.scope():
  model = TFBertForSequenceClassification.from_pretrained("klue/bert-base", num_labels=3, from_pt=True)
  optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5)
  model.compile(optimizer=optimizer, loss=model.hf_compute_loss, metrics = ['accuracy'])


# In[26]:


early_stopping = EarlyStopping(
    monitor="val_accuracy", 
    min_delta=0.001,
    patience=2)

model.fit(
    X_train, y_train, epochs=2, batch_size=32, validation_data = (X_val, y_val),
    callbacks = [early_stopping]
)


# In[ ]:


print("\n 테스트 정확도: %.4f" % (model.evaluate(X_test, y_test, batch_size=1024)[1]))

