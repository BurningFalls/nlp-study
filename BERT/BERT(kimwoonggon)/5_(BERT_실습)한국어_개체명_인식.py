#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/BurningFalls/nlp-study/blob/main/BERT%20(%EA%B9%80%EC%9B%85%EA%B3%A4)/5_(BERT_%EC%8B%A4%EC%8A%B5)%ED%95%9C%EA%B5%AD%EC%96%B4_%EA%B0%9C%EC%B2%B4%EB%AA%85_%EC%9D%B8%EC%8B%9D.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# # 한국어 개체명 인식 개요
# 이번 강의에서는 버트를 활용해서 한국어 개체명 인식을 다뤄보고자 합니다. 
# ![Imgur](https://i.imgur.com/PDdTLjy.png) 
# 데이터는 https://github.com/naver/nlp-challenge/ 에서 받아왔습니다.
# 
# 개체명 추출 리더보드에서 제공되는 코퍼스는 문장에 나타난 개체명을 14개 분류 카테고리로 주석 작업이 되어있습니다.
# 
# ![Imgur](https://i.imgur.com/it4uTE3.png)
# 
# 문장을 입력하면 다음과 같은 형식으로 개체명이 분류됩니다.  
# 
# ![Imgur](https://i.imgur.com/RpuZf6R.png)
# 
# 
# 

# # 목차
# 이번 실습은 <b>1) 네이버 개체명 인식 데이터 불러오기 및 전처리 2) BERT 인풋 만들기 3) 버트를 활용한 개체명 인식 모델 만들기 4) 훈련 및 성능 검증 5) 실제 데이터로 실습하기</b>로 구성되어 있습니다.

# # BERT를 활용하여 한국어 개체명 인식기 만들기

# ## 개체명 인식 데이터 불러오기 및 전처리

# 개체명 인식을 위한 데이터를 다운 받습니다.

# In[ ]:


get_ipython().system('wget https://github.com/naver/nlp-challenge/raw/master/missions/ner/data/train/train_data')


# 분석에 필요한 모듈들을 임포트 합니다.

# In[ ]:


get_ipython().system('pip install transformers')
get_ipython().system('pip install sentencepiece')
import tensorflow as tf
import numpy as np
import pandas as pd
from transformers import *
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
import os
import re
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split


# train 데이터를 불러 오겠습니다.

# In[ ]:


train = pd.read_csv("train_data", names=['src', 'tar'], sep="\t")
train = train.reset_index()
train


# train 데이터에 마침표가 이상한 것들이 많아서 확실하게 .으로 수정해 주겠습니다.

# In[ ]:


train['src'] = train['src'].str.replace("．", ".", regex=False)


# In[ ]:


train.loc[train['src']=='.']


# 데이터를 전처리 해주겠습니다.  
# 한글, 영어, 소문자, 대문자, . 이외의 단어들을 모두 제거하겠습니다.

# In[ ]:


train['src'] = train['src'].astype(str)
train['tar'] = train['tar'].astype(str)

train['src'] = train['src'].str.replace(r'[^ㄱ-ㅣ가-힣0-9a-zA-Z.]+', "", regex=True)


# 데이터를 리스트 형식으로 변환합니다.

# In[ ]:


data = [list(x) for x in train[['index', 'src', 'tar']].to_numpy()]


# 데이터를 잘 보면 (인덱스, 단어, 개체) 로 이루어 진 것을 알 수 있습니다.  
# 인덱스가 1,2,3,4,5.. 이렇게 이어지다가 다시 1,2,3,4, 로 바뀌는데 숫자가 바뀌기 전까지가 한 문장을 의미합니다.

# In[ ]:


print(data[:20])


# 라벨들을 추출하고, 딕셔너리 형식으로 저장하도록 하겠습니다.

# In[ ]:


label = train['tar'].unique().tolist()
label_dict = {word:i for i, word in enumerate(label)}
label_dict.update({"[PAD]":len(label_dict)})
index_to_ner = {i:j for j, i in label_dict.items()}


# In[ ]:


print(label_dict)


# In[ ]:


print(index_to_ner)


# 데이터를 문장들과 개체들로 분리합니다.  
# tups[0], tups[1],... 에 각각의 문장에 해당하는 단어와 개체 번호가 저장이 되게 됩니다.

# In[ ]:


tups = []
temp_tup = []
temp_tup.append(data[0][1:])
sentences = []
targets = []
for i, j, k in data:
  
  if i != 1:
    temp_tup.append([j,label_dict[k]])
  if i == 1:
    if len(temp_tup) != 0:
      tups.append(temp_tup)
      temp_tup = []
      temp_tup.append([j,label_dict[k]])

tups.pop(0)


# In[ ]:


print(tups[0], tups[1])


# tups를 보면 [(단어, 개체), (단어, 개체), (단어, 개체)]의 형식으로 저장이 되어 있는데, 이거를 (단어, 단어, 단어, 단어), (개체, 개체, 개체, 개체) 형식으로 변환하도록 하겠습니다.
# 

# In[ ]:


sentences = []
targets = []
for tup in tups:
  sentence = []
  target = []
  sentence.append("[CLS]")
  target.append(label_dict['-'])
  for i, j in tup:
    sentence.append(i)
    target.append(j)
  sentence.append("[SEP]")
  target.append(label_dict['-'])
  sentences.append(sentence)
  targets.append(target)


# In[ ]:


sentences[0]


# In[ ]:


targets[0]


# ## 버트 인풋 만들기

# 구글의 multilinguial-bert를 활용하도록 하겠습니다.

# In[ ]:


tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')


# In[ ]:


tokenizer.tokenize("대한민국 만세.")


# 여기서부터가 중요한데, 문장을 토크나이징 하고 개체(target)을 토크나이징 한 문장에 맞추도록 하겠습니다.  
# 문장 "대한민국 만세." 는 사실 (대한민국, 개체1), (만세., 개체2) 을 가지고 있는데 토크나이징을 하면 '▁대한민국', '▁만', '세', '.' 로 토크나이징이 됩니다.  
# 여기서 그렇다면 ( ▁대한민국, 개체1) , (▁만, 개체2), (세, 개체2), (., 개체 2) 와 같은 방식으로 각 개체를 부여해주어야 합니다.

# In[ ]:


def tokenize_and_preserve_labels(sentence, text_labels):
  tokenized_sentence = []
  labels = []

  for word, label in zip(sentence, text_labels):

    tokenized_word = tokenizer.tokenize(word)
    n_subwords = len(tokenized_word)

    tokenized_sentence.extend(tokenized_word)
    labels.extend([label] * n_subwords)

  return tokenized_sentence, labels


# In[ ]:


tokenized_texts_and_labels = [
                              tokenize_and_preserve_labels(sent, labs)
                              for sent, labs in zip(sentences, targets)]


# In[ ]:


print(tokenized_texts_and_labels[:2])
# [(문장, 개체들), (문장, 개체들),...] 형식으로 저장되어 있음.


# (문장, 개체들), (문장, 개체들) 을 [문장, 문장, 문장, ...] , [개체들, 개체들 개체들,,,,]로 분리해주도록 하겠습니다.

# In[ ]:


tokenized_texts = [token_label_pair[0] for token_label_pair in tokenized_texts_and_labels]
labels = [token_label_pair[1] for token_label_pair in tokenized_texts_and_labels]


# In[ ]:


tokenized_texts[1]


# In[ ]:


labels[1]


# 문장의 길이가 상위 2.5%(88) 인 지점을 기준으로 문장의 길이를 정하도록 하겠습니다.  
# 만약 문장의 길이가 88보다 크면 문장이 잘리게 되고, 길이가 88보다 작다면 패딩이 되어 모든 문장의 길이가 88로 정해지게 됩니다.

# In[ ]:


print(np.quantile(np.array([len(x) for x in tokenized_texts]), 0.975))
max_len = 88
bs = 32


# 버트에 인풋으로 들어갈 train 데이터를 만들도록 하겠습니다.  
# 버트 인풋으로는   
# input_ids : 문장이 토크나이즈 된 것이 숫자로 바뀐 것,   
# attention_masks : 문장이 토크나이즈 된 것 중에서 패딩이 아닌 부분은 1, 패딩인 부분은 0으로 마스킹  
# [input_ids, attention_masks]가 인풋으로 들어갑니다.

# In[ ]:


input_ids = pad_sequences([tokenizer.convert_tokens_to_ids(txt) for txt in tokenized_texts],
                          maxlen=max_len, dtype = "int", value=tokenizer.convert_tokens_to_ids("[PAD]"), truncating="post", padding="post")


# In[ ]:


input_ids[1]


# 정답에 해당하는 개체들을 만들어 보겠습니다.  
# 패딩에 해당하는 부분은 label_dict([PAD])(29)가 들어가게 되겠습니다.

# In[ ]:


tags = pad_sequences([lab for lab in labels], maxlen=max_len, value=label_dict["[PAD]"], padding='post',                     dtype='int', truncating='post')


# In[ ]:


tags[1]


# 어텐션 마스크를 만들어 주겠습니다.

# In[ ]:


attention_masks = np.array([[int(i != tokenizer.convert_tokens_to_ids("[PAD]")) for i in ii] for ii in input_ids])


# train 데이터에서 10% 만큼을 validation 데이터로 분리해 주겠습니다.

# In[ ]:


tr_inputs, val_inputs, tr_tags, val_tags = train_test_split(input_ids, tags,
                                                            random_state=2018, test_size=0.1)


# In[ ]:


tr_masks, val_masks, _, _ = train_test_split(attention_masks, input_ids,
                                             random_state=2018, test_size=0.1)


# ## 개체명 인식 모델 만들기

# In[ ]:


# TPU 작동을 위해 실행
resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu='grpc://' + os.environ['COLAB_TPU_ADDR'])
tf.config.experimental_connect_to_cluster(resolver)
tf.tpu.experimental.initialize_tpu_system(resolver)


# In[ ]:


SEQ_LEN = max_len
def create_model():
  model = TFBertModel.from_pretrained("bert-base-multilingual-cased", from_pt=True, num_labels=len(label_dict), output_attentions = False,
    output_hidden_states = False)

  token_inputs = tf.keras.layers.Input((SEQ_LEN,), dtype=tf.int32, name='input_word_ids') # 토큰 인풋
  mask_inputs = tf.keras.layers.Input((SEQ_LEN,), dtype=tf.int32, name='input_masks') # 마스크 인풋

  bert_outputs = model([token_inputs, mask_inputs])
  bert_outputs = bert_outputs[0] # shape : (Batch_size, max_len, 30(개체의 총 개수))
  nr = tf.keras.layers.Dense(30, activation='softmax')(bert_outputs) # shape : (Batch_size, max_len, 30)
  
  nr_model = tf.keras.Model([token_inputs, mask_inputs], nr)
  
  nr_model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.00002), loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
      metrics=['sparse_categorical_accuracy'])
  nr_model.summary()
  return nr_model


# ## 훈련 및 성능 검증

# In[ ]:


strategy = tf.distribute.experimental.TPUStrategy(resolver)
# TPU를 활용하기 위해 context로 묶어주기
with strategy.scope():
  nr_model = create_model()
  nr_model.fit([tr_inputs, tr_masks], tr_tags, validation_data=([val_inputs, val_masks], val_tags), epochs=3, shuffle=False, batch_size=bs)


# In[ ]:


# 만약 TPU를 사용하지 않고 GPU를 사용한다면
#nr_model = create_model()
#nr_model.fit([tr_inputs, tr_masks], tr_tags, validation_data=([val_inputs, val_masks], val_tags), epochs=3, shuffle=False, batch_size=bs)


# In[ ]:


from sklearn.metrics import precision_score, recall_score, f1_score, classification_report


# In[ ]:


y_predicted = nr_model.predict([val_inputs, val_masks])


# In[ ]:


f_label = [i for i, j in label_dict.items()]
val_tags_l = [index_to_ner[x] for x in np.ravel(val_tags).astype(int).tolist()]
y_predicted_l = [index_to_ner[x] for x in np.ravel(np.argmax(y_predicted, axis=2)).astype(int).tolist()]
f_label.remove("[PAD]")


# 각 개체별 f1 score를 측정하도록 하겠습니다.  
# 참고로 micro avg는 전체 정답을 기준으로 f1 score을 측정한 것이며,  
# macro avg는 각 개체별 f1 score를 가중평균 한 것입니다.

# In[ ]:


print(classification_report(val_tags_l, y_predicted_l, labels=f_label))


# # 실제 데이터로 실습하기

# In[ ]:


def ner_inference(test_sentence):
  

  tokenized_sentence = np.array([tokenizer.encode(test_sentence, max_length=max_len, truncation=True, padding='max_length')])
  tokenized_mask = np.array([[int(x!=1) for x in tokenized_sentence[0].tolist()]])
  ans = nr_model.predict([tokenized_sentence, tokenized_mask])
  ans = np.argmax(ans, axis=2)

  tokens = tokenizer.convert_ids_to_tokens(tokenized_sentence[0])
  new_tokens, new_labels = [], []
  for token, label_idx in zip(tokens, ans[0]):
    if (token.startswith("##")):
      new_labels.append(index_to_ner[label_idx])
      new_tokens.append(token[2:])
    elif (token=='[CLS]'):
      pass
    elif (token=='[SEP]'):
      pass
    elif (token=='[PAD]'):
      pass
    elif (token != '[CLS]' or token != '[SEP]'):
      new_tokens.append(token)
      new_labels.append(index_to_ner[label_idx])

  for token, label in zip(new_tokens, new_labels):
      print("{}\t{}".format(label, token))


# In[ ]:


ner_inference("문재인 대통령은 1953년 1월 24일 경상남도 거제시에서 아버지 문용형과 어머니 강한옥 사이에서 둘째(장남)로 태어났다.")


# In[ ]:


ner_inference("9세이브로 구완 30위인 LG 박찬형은 평균자책점이 16.45로 준수한 편이지만 22이닝 동안 피홈런이 31개나 된다.")


# In[ ]:


ner_inference("인공지능의 역사는 20세기 초반에서 더 거슬러 올라가보면 이미 17~18세기부터 태동하고 있었지만 이때는 인공지능 그 자체보다는 뇌와 마음의 관계에 관한 철학적인 논쟁 수준에 머무르고 있었다. 그럴 수 밖에 없는 것이 당시에는 인간의 뇌 말고는 정보처리기계가 존재하지 않았기 때문이다. ")

