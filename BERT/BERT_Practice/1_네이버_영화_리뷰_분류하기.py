#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/BurningFalls/nlp-study/blob/main/BERT%20%EC%8B%A4%EC%8A%B5/BERT_%EB%84%A4%EC%9D%B4%EB%B2%84_%EC%98%81%ED%99%94_%EB%A6%AC%EB%B7%B0_%EB%B6%84%EB%A5%98%ED%95%98%EA%B8%B0.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

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
from transformers import BertTokenizer, TFBertModel


# In[3]:


urllib.request.urlretrieve("https://raw.githubusercontent.com/e9t/nsmc/master/ratings_train.txt", filename="ratings_train.txt")
urllib.request.urlretrieve("https://raw.githubusercontent.com/e9t/nsmc/master/ratings_test.txt", filename="ratings_test.txt")


# In[4]:


train_data = pd.read_table('ratings_train.txt')
test_data = pd.read_table('ratings_test.txt')


# In[5]:


# delete null values
train_data = train_data.dropna(how='any')
train_data = train_data.reset_index(drop=True)
assert train_data.isnull().values.any() == False

test_data = test_data.dropna(how='any')
test_data = test_data.reset_index(drop=True)
assert test_data.isnull().values.any() == False


# In[6]:


max_seq_len = 128
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')


# In[7]:


def convert_examples_to_features(examples, labels):
  input_ids = []        # word embedding을 위한 문장의 정수 인코딩
  attention_masks = []  # 실제 단어가 있는 곳은 1, 아니면 0
  token_type_ids = []   # segment embedding: 0(문장 1개), 0 and 1(문장 2개)
  data_labels = []
  
  for example in tqdm(examples, total=len(examples)):
    input_id = tokenizer.encode(example, max_length=max_seq_len, pad_to_max_length=True)

    padding_count = input_id.count(tokenizer.pad_token_id)
    attention_mask = [1] * (max_seq_len - padding_count) + [0] * padding_count

    token_type_id = [0] * max_seq_len

    assert len(input_id) == max_seq_len
    assert len(attention_mask) == max_seq_len
    assert len(token_type_id) == max_seq_len

    input_ids.append(input_id)
    attention_masks.append(attention_mask)
    token_type_ids.append(token_type_id)

  if labels is not None:
    for label in labels:
      data_labels.append(label)

  input_ids = np.array(input_ids, dtype=int)
  attention_masks = np.array(attention_masks, dtype=int)
  token_type_ids = np.array(token_type_ids, dtype=int)
  data_labels = np.asarray(data_labels, dtype=np.int32)

  return (input_ids, attention_masks, token_type_ids), data_labels


# In[8]:


train_X, train_y = convert_examples_to_features(train_data['document'], train_data['label'])


# In[9]:


test_X, test_y = convert_examples_to_features(test_data['document'], test_data['label'])


# In[10]:


class TFBertForSequenceClassification(tf.keras.Model):
  def __init__(self, model_name):
    super(TFBertForSequenceClassification, self).__init__()
    self.bert = TFBertModel.from_pretrained(model_name, from_pt=True)
    self.classifier = tf.keras.layers.Dense(1,
                                            kernel_initializer=tf.keras.initializers.TruncatedNormal(0.02),
                                            activation='sigmoid',
                                            name='classifier')
    
  def call(self, inputs):
    input_ids, attention_mask, token_type_ids = inputs
    outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
    cls_token = outputs[1]
    prediction = self.classifier(cls_token)

    return prediction


# In[11]:


resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu='grpc://' + os.environ['COLAB_TPU_ADDR'])

tf.config.experimental_connect_to_cluster(resolver)
tf.tpu.experimental.initialize_tpu_system(resolver)

strategy = tf.distribute.TPUStrategy(resolver)


# In[12]:


with strategy.scope():
  model = TFBertForSequenceClassification("bert-base-multilingual-cased")
  optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5)
  loss = tf.keras.losses.BinaryCrossentropy()
  model.compile(optimizer=optimizer, loss=loss, metrics = ['accuracy'])


# In[13]:


model.fit(train_X, train_y, epochs=2, batch_size=64, validation_split=0.2)


# In[14]:


results = model.evaluate(test_X, test_y, batch_size=1024)
print(f'test loss: {results[0] :.6f}')
print(f'test accuracy: {results[1] :.6f}')


# In[15]:


def sentiment_predict(new_sentence):
  input_id = tokenizer.encode(new_sentence, max_length=max_seq_len, pad_to_max_length=True)

  padding_count = input_id.count(tokenizer.pad_token_id)
  attention_mask = [1] * (max_seq_len - padding_count) + [0] * padding_count
  token_type_id = [0] * max_seq_len

  input_ids = np.array([input_id])
  attention_masks = np.array([attention_mask])
  token_type_ids = np.array([token_type_id])

  encoded_input = [input_ids, attention_masks, token_type_ids]
  score = model.predict(encoded_input)[0][0]
  print(f'score: {score :.6f}')

  if score > 0.5:
    print(f'{score * 100 :.2f}% 확률로 긍정 리뷰입니다.')
  elif score <= 0.5:
    print(f'{(1 - score) * 100 :.2f}% 확률로 부정 리뷰입니다.')


# In[16]:


sentiment_predict("보던거라 계속보고있는데 전개도 느리고 주인공인 은희는 한두컷 나오면서 소극적인모습에 ")


# In[17]:


sentiment_predict("스토리는 확실히 실망이였지만 배우들 연기력이 대박이였다 특히 이제훈 연기 정말 ... 이 배우들로 이렇게밖에 만들지 못한 영화는 아쉽지만 배우들 연기력과 사운드는 정말 빛났던 영화. 기대하고 극장에서 보면 많이 실망했겠지만 평점보고 기대없이 집에서 편하게 보면 괜찮아요. 이제훈님 연기력은 최고인 것 같습니다")


# In[18]:


sentiment_predict("별 똥같은 영화를 다 보네. 개별로입니다.")


# In[19]:


sentiment_predict("이 영화 존잼입니다 대박.")


# In[20]:


sentiment_predict('이 영화 개꿀잼 ㅋㅋㅋ')


# In[21]:


sentiment_predict('이딴게 영화냐 ㅉㅉ')


# In[22]:


sentiment_predict('감독 뭐하는 놈이냐?')


# In[23]:


sentiment_predict('와 개쩐다 정말 세계관 최강자들의 영화다')

