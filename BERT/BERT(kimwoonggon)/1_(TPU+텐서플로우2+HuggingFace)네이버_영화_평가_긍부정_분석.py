#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/BurningFalls/nlp-study/blob/main/BERT%20(%EA%B9%80%EC%9B%85%EA%B3%A4)/1_(TPU%2B%ED%85%90%EC%84%9C%ED%94%8C%EB%A1%9C%EC%9A%B02%2BHuggingFace)%EB%84%A4%EC%9D%B4%EB%B2%84_%EC%98%81%ED%99%94_%ED%8F%89%EA%B0%80_%EA%B8%8D%EB%B6%80%EC%A0%95_%EB%B6%84%EC%84%9D.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# # Huggingface 소개
# 
# ![대체 텍스트](https://i.imgur.com/6DM516z.png)  
# ![대체 텍스트](https://i.imgur.com/g6R67h4.png)    
# https://huggingface.co/transformers/ 
# 
# HuggingFace는 자연어 처리 인공지능 모델에서, BERT 모델 같은 트랜스포머 모델들을 쉽게 다룰 수 있게 해주는 패키지입니다.  
# 기본적으로 pytorch 기반으로 만들어져 있지만, 텐서플로우 2.0에서도 본 패키지를 사용 가능합니다.  
# 텐서플로우 2.0은 기존 케라스를 포함하고 있기 때문에, 기존 텐서플로우나 케라스에 익숙하신 분들이 쉽게 사용할 수 있습니다.  
# 텐서플로우 2.0 기반의 huggingface 사용 방법을 네이버 영화 긍부정 분석을 실습하면서 배워 보도록 하겠습니다.

# #목차
# 이번 실습은 <b>1) 네이버 감성분석 데이터 불러오기 및 전처리 2) BERT 인풋 만들기 3) 버트를 활용한 감성분석 모델 만들기 4) 훈련 및 성능 검증 5) 실제 데이터로 실습하기</b>로 구성되어 있습니다.

# #BERT를 활용하여 네이버 감성분석 만들기

# ## 네이버 감성분석 데이터 불러오기 및 전처리

# huggingface 패키지를 Colab에 설치합니다

# In[ ]:


get_ipython().system('pip install transformers')


# 텐서플로우 2와 필요한 모듈들을 임포트합니다.  
# 최근에 텐서플로우 기본 버전은 2로 바뀌었습니다.

# In[ ]:


import tensorflow as tf
import numpy as np
import pandas as pd
from transformers import *
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
import os


# 구글 드라이브와 Colab을 연동합니다.

# In[ ]:


import os
from google.colab import drive
drive.mount('/content/gdrive/')


# In[ ]:


os.listdir('gdrive/MyDrive/bert_google_research')


# 이번 예제에서 사용할 네이버 영화 감상분석 데이터를 다운로드 합니다

# In[ ]:


# 네이버 영화 감성분석 데이터 다운로드
get_ipython().system('git clone https://github.com/e9t/nsmc.git')


# In[ ]:


os.listdir('nsmc')


# 딥러닝 훈련에 사용 할 train 데이터와 test 데이터를 pandas dataframe 형식으로 불러옵니다.

# In[ ]:


train = pd.read_table("nsmc/"+"ratings_train.txt")
test = pd.read_table("nsmc/"+"ratings_test.txt")


# In[ ]:


train[50:70]


# ## 버트 인풋 만들기

# 한글 데이터를 분석하려면, 100개가 넘는 언어에 대해 훈련된 버트를 사용해야 합니다.  
# multilingual BERT를 사용하도록 하겠습니다.  
# 모델을 로드하기에 앞서, 토크나이저를 불러오도록 하겠습니다.  
# huggingface에서는 아주 쉽게 토크나이저를 불러올 수 있습니다.

# In[ ]:


tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')


# 버트를 사용하기에 앞서 가장 기초에 속하는 tokenizer 사용 방법에 대해서 잠시 배워보도록 하겠습니다.  
# tokenizer.encode => 문장을 버트 모델의 인풋 토큰값으로 바꿔줌  
# tokenizer.tokenize => 문장을 토큰화

# In[ ]:


print(tokenizer.encode("보는내내 그대로 들어맞는 예측 카리스마 없는 악역"))


# In[ ]:


print(tokenizer.tokenize("보는내내 그대로 들어맞는 예측 카리스마 없는 악역"))


# 우리가 네이버 영화 평가 긍부정 분석을 위해, train 15만개의 데이터를 버트의 인풋 값으로 바꿔주겠습니다.  
# 버트의 인풋은 토큰, 세그멘트, 마스크로 나눠집니다.  
# 이 세 값이 버트 모형에 들어가서, 버트 모형에 맞게 고차원으로 임베딩이 되게 되는 원리입니다.  
# 
# 토큰은 말 그대로 단어를 단어사전의 위치값으로 표현해주는 것이며, 
# 세그멘트는 버트 모형에서 문장이 앞 문장인지, 뒷 문장인지 표현해주는 것입니다.(본 예제는 인풋으로 문장이 하나만 들어가므로 0으로 통일)  
# 마스크는 문장이 유효한 값인지, 아니면 유효하지 않은 값이라 패딩 값으로 채운 것인지를 나타냅니다.  
# 문장이 유효한 값이면 1로 채우고, 유효하지 않은 값이면 0으로 채우게 됩니다.  
# 문장마다 문장 길이는 다르지만, 버트의 인풋 길이는 일정해야 하므로, 버트에서 지정한 문장 길이를 초과하면 패딩값인 0을 채우게 됩니다.

# In[ ]:


print(tokenizer.tokenize("전율을 일으키는 영화. 다시 보고싶은 영화"))


# In[ ]:


print(tokenizer.encode("전율을 일으키는 영화. 다시 보고싶은 영화"))


# In[ ]:


print(tokenizer.encode("전율을 일으키는 영화. 다시 보고싶은 영화", max_length=128, pad_to_max_length=True))


# 토큰 인풋의 예를 들면 다음과 같습니다.  
# 문장을 토크나이징 하면 "전율을 일으키는 영화. 다시 보고싶은 영화"가  
# "'전', '##율', '##을', '일', '##으', '##키는', '영화', '.', '다시', '보고', '##싶', '##은', '영화'" 로 토크나이징이 됩니다.  
# 이거를 버트 인풋에 들어갈 숫자로 바꿔주면,  
# ["101, 9665, 119183, 10622, 9641, 119185, 66815, 42428, 119, 25805, 98199, 119088, 10892, 42428, 102"]  
# 로 바뀌게 됩니다. 여기 나오는 숫자들이 버트 인풋에 들어가는 토큰 인풋입니다.  
# 버트 모형에 들어가는 인풋은 사실 일정한 길이를 가져야 합니다.(본 예제에서는 128)  
# 따라서 남는 부분은 0으로 채워지게 됩니다(패딩)

# In[ ]:


# 세그멘트 인풋
print([0]*128)


# 세그멘트 인풋은 문장이 앞문장인지 뒷문장인지 구분해주는 역할을 하는데요  
# 본 문장에서는 문장 하나만 인풋으로 들어가기 때문에 0만 들어가게 되고, 문장 길이만큼의 0이 인풋으로 들어가게 됩니다.

# In[ ]:


# 마스크 인풋
valid_num = len(tokenizer.encode("전율을 일으키는 영화. 다시 보고싶은 영화"))
print(valid_num * [1] + (128 - valid_num) * [0])


# 마스크 인풋은 토큰 인풋에서 패딩이 아닌 부분은 1, 패딩인 부분은 0으로 두게 됩니다.

# 종합하면,  
# 버트의 인풋은 토큰, 세그먼트, 마스크로 이루어져 있습니다.  
# "전율을 일으키는 영화. 다시 보고싶은 영화" 라는 문장을 가지고 예를 들면,
# 
# 토큰 인풋 : [101, 9665, 119183, 10622, 9641, 119185, 66815, 42428, 119, 25805, 98199, 119088, 10892, 42428, 102, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
# 
# 세그먼트 인풋 : [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
# 
# 마스크 인풋 : [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

# 네이버 영화 평가 문장들을 버트 인풋으로 바꿔보도록 하겠습니다.  
# 문장이 토큰 인풋, 세그먼트 인풋, 마스크 인풋으로 변환 됩니다.  
# huggingface에서는 순서가 [토큰 인풋, 마스크 인풋, 세그먼트 인풋] 입니다.

# In[ ]:


def convert_data(data_df):
    global tokenizer
    
    SEQ_LEN = 128 #SEQ_LEN : 버트에 들어갈 인풋의 길이
    
    tokens, masks, segments, targets = [], [], [], []
    
    for i in tqdm(range(len(data_df))):
        # token : 문장을 토큰화함
        token = tokenizer.encode(data_df[DATA_COLUMN][i], max_length=SEQ_LEN, truncation=True, padding='max_length')
       
        # 마스크는 토큰화한 문장에서 패딩이 아닌 부분은 1, 패딩인 부분은 0으로 통일
        num_zeros = token.count(0)
        mask = [1]*(SEQ_LEN-num_zeros) + [0]*num_zeros
        
        # 문장의 전후관계를 구분해주는 세그먼트는 문장이 1개밖에 없으므로 모두 0
        segment = [0]*SEQ_LEN

        # 버트 인풋으로 들어가는 token, mask, segment를 tokens, segments에 각각 저장
        tokens.append(token)
        masks.append(mask)
        segments.append(segment)
        
        # 정답(긍정 : 1 부정 0)을 targets 변수에 저장해 줌
        targets.append(data_df[LABEL_COLUMN][i])

    # tokens, masks, segments, 정답 변수 targets를 numpy array로 지정    
    tokens = np.array(tokens)
    masks = np.array(masks)
    segments = np.array(segments)
    targets = np.array(targets)

    return [tokens, masks, segments], targets

# 위에 정의한 convert_data 함수를 불러오는 함수를 정의
def load_data(pandas_dataframe):
    data_df = pandas_dataframe
    data_df[DATA_COLUMN] = data_df[DATA_COLUMN].astype(str)
    data_df[LABEL_COLUMN] = data_df[LABEL_COLUMN].astype(int)
    data_x, data_y = convert_data(data_df)
    return data_x, data_y

SEQ_LEN = 128
BATCH_SIZE = 20
# 긍부정 문장을 포함하고 있는 칼럼
DATA_COLUMN = "document"
# 긍정인지 부정인지를 (1=긍정,0=부정) 포함하고 있는 칼럼
LABEL_COLUMN = "label"

# train 데이터를 버트 인풋에 맞게 변환
train_x, train_y = load_data(train)


# In[ ]:


# 훈련 성능을 검증한 test 데이터를 버트 인풋에 맞게 변환
test_x, test_y = load_data(test)


# ## 버트를 활용한 감성분석 모델 만들기

# 버트 훈련을 빠르게 하기 위해, TPU를 사용하도록 하겠습니다.  
# TPU를 사용하시고 싶지 않으신 분은 그냥 TPU 관련 부분을 실행하지 않으면 되겠습니다.

# In[ ]:


# TPU 객체 지정
TPU = True
if TPU:
  resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu='grpc://' + os.environ['COLAB_TPU_ADDR'])
  tf.config.experimental_connect_to_cluster(resolver)
  tf.tpu.experimental.initialize_tpu_system(resolver)
else:
  pass


# In[ ]:


model = TFBertModel.from_pretrained('bert-base-multilingual-cased')
# 토큰 인풋, 마스크 인풋, 세그먼트 인풋 정의
token_inputs = tf.keras.layers.Input((SEQ_LEN,), dtype=tf.int32, name='input_word_ids')
mask_inputs = tf.keras.layers.Input((SEQ_LEN,), dtype=tf.int32, name='input_masks')
segment_inputs = tf.keras.layers.Input((SEQ_LEN,), dtype=tf.int32, name='input_segment')
# 인풋이 [토큰, 마스크, 세그먼트]인 모델 정의
bert_outputs = model([token_inputs, mask_inputs, segment_inputs])
# 버트 아웃풋의 텐서의 shape은 [batch_size, 문장의 길이, 768]임


# In[ ]:


bert_outputs = bert_outputs[1]
sentiment_first = tf.keras.layers.Dense(1, activation='sigmoid', kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02))(bert_outputs)
sentiment_model = tf.keras.Model([token_inputs, mask_inputs, segment_inputs], sentiment_first)
sentiment_model.compile(optimizer=tf.keras.optimizers.Adam(lr=1.0e-5), loss=tf.keras.losses.BinaryCrossentropy(), metrics = ['accuracy'])


# In[ ]:


sentiment_model.summary()


# 감정 분석에 맞는 버트 모형을 리턴하는 함수를 정의하도록 하겠습니다.  
# TPU를 활용하려면 함수로 묶어야 합니다.

# In[ ]:


# Rectified Adam 옵티마이저 사용
get_ipython().system('pip install tensorflow_addons')
import tensorflow_addons as tfa
opt = tfa.optimizers.RectifiedAdam(lr=1.0e-5, weight_decay=0.0025, warmup_proportion=0.05)


# In[ ]:


def create_sentiment_bert():
  # 버트 pretrained 모델 로드
  model = TFBertModel.from_pretrained('bert-base-multilingual-cased')
  # 토큰 인풋, 마스크 인풋, 세그먼트 인풋 정의
  token_inputs = tf.keras.layers.Input((SEQ_LEN,), dtype=tf.int32, name='input_word_ids')
  mask_inputs = tf.keras.layers.Input((SEQ_LEN,), dtype=tf.int32, name='input_masks')
  segment_inputs = tf.keras.layers.Input((SEQ_LEN,), dtype=tf.int32, name='input_segment')
  # 인풋이 [토큰, 마스크, 세그먼트]인 모델 정의
  bert_outputs = model([token_inputs, mask_inputs, segment_inputs])

  bert_outputs = bert_outputs[1]
  sentiment_first = tf.keras.layers.Dense(1, activation='sigmoid', kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02))(bert_outputs)
  sentiment_model = tf.keras.Model([token_inputs, mask_inputs, segment_inputs], sentiment_first)

  sentiment_model.compile(optimizer=opt, loss=tf.keras.losses.BinaryCrossentropy(), metrics = ['accuracy'])
  return sentiment_model


# ## 훈련 및 성능 검증

# In[ ]:


# TPU 실행 시
if TPU:
  strategy = tf.distribute.experimental.TPUStrategy(resolver)
# 함수를 strategy.scope로 묶어 줌
  with strategy.scope():
    sentiment_model = create_sentiment_bert()
  
  sentiment_model.fit(train_x, train_y, epochs=4, shuffle=True, batch_size=100, validation_data=(test_x, test_y))
else:
  # GPU 모드로 훈련시킬 때
  sentiment_model = create_sentiment_bert()
  
  sentiment_model.fit(train_x, train_y, epochs=4, shuffle=True, batch_size=100, validation_data=(test_x, test_y))


# 훈련한 모델을 path에 저장
# path는 임의로 지정해 주세요

# In[ ]:


# PATH는 임의로 지정

path = "gdrive/My Drive/bert_google_research/naver_sentiment"


# In[ ]:


sentiment_model.save_weights(path+"/huggingface_bert.h5")


# 훈련 모델의 예측 성능을 F1 SCORE로 체크하기 위한 작업

# In[ ]:


def predict_convert_data(data_df):
    global tokenizer
    tokens, masks, segments = [], [], []
    
    for i in tqdm(range(len(data_df))):

        token = tokenizer.encode(data_df[DATA_COLUMN][i], max_length=SEQ_LEN, truncation=True, padding='max_length')
        num_zeros = token.count(0)
        mask = [1]*(SEQ_LEN-num_zeros) + [0]*num_zeros
        segment = [0]*SEQ_LEN

        tokens.append(token)
        segments.append(segment)
        masks.append(mask)

    tokens = np.array(tokens)
    masks = np.array(masks)
    segments = np.array(segments)
    return [tokens, masks, segments]

# 위에 정의한 convert_data 함수를 불러오는 함수를 정의
def predict_load_data(pandas_dataframe):
    data_df = pandas_dataframe
    data_df[DATA_COLUMN] = data_df[DATA_COLUMN].astype(str)
    data_x = predict_convert_data(data_df)
    return data_x


# test 데이터 예측하기

# In[ ]:


test_set = predict_load_data(test)


# In[ ]:


test_set


# In[ ]:


#TPU를 사용하기 위해서
with strategy.scope():
  preds = sentiment_model.predict(test_set)


# In[ ]:


# 부정이면 0, 긍정이면 1 출력
preds


# 우리가 훈련한 모델을 F1 SCORE를 바탕으로 성능 측정  
# F1 SCORE는 precision과 recall을 가중평균하여 계산합니다  
# recall은 (모델이 TRUE라고 판정한 것의 숫자)/(전체 TRUE의 숫자)  
# precision은 (진짜 TRUE) / (모델이 TRUE라고 판정한 것의 숫자)

# In[ ]:


from sklearn.metrics import classification_report
y_true = test['label']
# F1 Score 확인
print(classification_report(y_true, np.round(preds,0)))


# In[ ]:


import logging
tf.get_logger().setLevel(logging.ERROR)


# # 실제 데이터로 실습하기

# 문장 하나 하나를 가지고 실제로 분류해보도록 하겠습니다.  

# In[ ]:


def sentence_convert_data(data):
    global tokenizer
    tokens, masks, segments = [], [], []
    token = tokenizer.encode(data, max_length=SEQ_LEN, truncation=True, padding='max_length')
    
    num_zeros = token.count(0) 
    mask = [1]*(SEQ_LEN-num_zeros) + [0]*num_zeros 
    segment = [0]*SEQ_LEN

    tokens.append(token)
    segments.append(segment)
    masks.append(mask)

    tokens = np.array(tokens)
    masks = np.array(masks)
    segments = np.array(segments)
    return [tokens, masks, segments]

def movie_evaluation_predict(sentence):
    data_x = sentence_convert_data(sentence)
    predict = sentiment_model.predict(data_x)
    predict_value = np.ravel(predict)
    predict_answer = np.round(predict_value,0).item()
    
    if predict_answer == 0:
      print("(부정 확률 : %.2f) 부정적인 영화 평가입니다." % (1-predict_value))
    elif predict_answer == 1:
      print("(긍정 확률 : %.2f) 긍정적인 영화 평가입니다." % predict_value)


# In[ ]:


movie_evaluation_predict("보던거라 계속보고있는데 전개도 느리고 주인공인 은희는 한두컷 나오면서 소극적인모습에 ")


# In[ ]:


movie_evaluation_predict("스토리는 확실히 실망이였지만 배우들 연기력이 대박이였다 특히 이제훈 연기 정말 ... 이 배우들로 이렇게밖에 만들지 못한 영화는 아쉽지만 배우들 연기력과 사운드는 정말 빛났던 영화. 기대하고 극장에서 보면 많이 실망했겠지만 평점보고 기대없이 집에서 편하게 보면 괜찮아요. 이제훈님 연기력은 최고인 것 같습니다")


# In[ ]:


movie_evaluation_predict("남친이 이 영화를 보고 헤어지자고한 영화. 자유롭게 살고 싶다고 한다. 내가 무슨 나비를 잡은 덫마냥 나에겐 다시 보고싶지 않은 영화.")

