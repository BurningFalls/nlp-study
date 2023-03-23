#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/BurningFalls/nlp-study/blob/main/BERT%20%EC%8B%A4%EC%8A%B5/BERT_KoBERT%EB%A5%BC_%EC%9D%B4%EC%9A%A9%ED%95%9C_%EA%B0%9C%EC%B2%B4%EB%AA%85_%EC%9D%B8%EC%8B%9D(Named_Entity_Recognition).ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# In[1]:


pip install transformers


# In[2]:


pip install seqeval


# # 1. 데이터 로드
# 

# In[3]:


import pandas as pd
import numpy as np
import os
from tqdm import tqdm
from transformers import shape_list, BertTokenizer, TFBertModel
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.sequence import pad_sequences
from seqeval.metrics import f1_score, classification_report
import tensorflow as tf


# In[4]:


get_ipython().system('wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate \'https://docs.google.com/uc?export=download&id=1jatBP8yZkWn6Kg6mjN7nWLnYVwXE_sY_\' -O- | sed -rn \'s/.*confirm=([0-9A-Za-z_]+).*/\\1\\n/p\')&id=1jatBP8yZkWn6Kg6mjN7nWLnYVwXE_sY_" -O ner_train_data.csv && rm -rf /tmp/cookies.txt')


# In[5]:


get_ipython().system('wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate \'https://docs.google.com/uc?export=download&id=1YVYShKCtWfigXBOb5ie7s6QmA-dHnjt3\' -O- | sed -rn \'s/.*confirm=([0-9A-Za-z_]+).*/\\1\\n/p\')&id=1YVYShKCtWfigXBOb5ie7s6QmA-dHnjt3" -O ner_test_data.csv && rm -rf /tmp/cookies.txt')


# In[6]:


get_ipython().system('wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate \'https://docs.google.com/uc?export=download&id=1_DPfdY1Q5Xt2md7QVbKcQLRDYHm3qgVQ\' -O- | sed -rn \'s/.*confirm=([0-9A-Za-z_]+).*/\\1\\n/p\')&id=1_DPfdY1Q5Xt2md7QVbKcQLRDYHm3qgVQ" -O ner_label.txt && rm -rf /tmp/cookies.txt')


# In[7]:


train_ner_df = pd.read_csv("ner_train_data.csv")


# In[8]:


train_ner_df.head()


# In[9]:


test_ner_df = pd.read_csv("ner_test_data.csv")


# In[10]:


test_ner_df.head()


# In[11]:


train_data_sentence = [sent.split() for sent in train_ner_df['Sentence'].values]
test_data_sentence = [sent.split() for sent in test_ner_df['Sentence'].values]
train_data_label = [tag.split() for tag in train_ner_df['Tag'].values]
test_data_label = [tag.split() for tag in test_ner_df['Tag'].values]


# In[12]:


labels = [label.strip() for label in open('ner_label.txt', 'r', encoding='utf-8')]
print('개체명 태깅 정보 :', labels)


# In[13]:


tag_to_index = {tag: index for index, tag in enumerate(labels)}
index_to_tag = {index: tag for index, tag in enumerate(labels)}


# In[14]:


tag_size = len(tag_to_index)
print('개체명 태깅 정보의 개수 :',tag_size)


# In[15]:


tokenizer = BertTokenizer.from_pretrained("klue/bert-base")


# In[16]:


def convert_examples_to_features(examples, labels, max_seq_len, tokenizer,
                                 pad_token_id_for_segment=0, pad_token_id_for_label=-100):
    cls_token = tokenizer.cls_token
    sep_token = tokenizer.sep_token
    pad_token_id = tokenizer.pad_token_id

    input_ids, attention_masks, token_type_ids, data_labels = [], [], [], []

    for example, label in tqdm(zip(examples, labels), total=len(examples)):
        tokens = []
        labels_ids = []
        for one_word, label_token in zip(example, label):
            subword_tokens = tokenizer.tokenize(one_word)
            tokens.extend(subword_tokens)
            labels_ids.extend([tag_to_index[label_token]]+ [pad_token_id_for_label] * (len(subword_tokens) - 1))

        special_tokens_count = 2
        if len(tokens) > max_seq_len - special_tokens_count:
            tokens = tokens[:(max_seq_len - special_tokens_count)]
            labels_ids = labels_ids[:(max_seq_len - special_tokens_count)]

        tokens += [sep_token]
        labels_ids += [pad_token_id_for_label]
        tokens = [cls_token] + tokens
        labels_ids = [pad_token_id_for_label] + labels_ids

        input_id = tokenizer.convert_tokens_to_ids(tokens)
        attention_mask = [1] * len(input_id)
        padding_count = max_seq_len - len(input_id)

        input_id = input_id + ([pad_token_id] * padding_count)
        attention_mask = attention_mask + ([0] * padding_count)
        token_type_id = [pad_token_id_for_segment] * max_seq_len
        label = labels_ids + ([pad_token_id_for_label] * padding_count)

        assert len(input_id) == max_seq_len, "Error with input length {} vs {}".format(len(input_id), max_seq_len)
        assert len(attention_mask) == max_seq_len, "Error with attention mask length {} vs {}".format(len(attention_mask), max_seq_len)
        assert len(token_type_id) == max_seq_len, "Error with token type length {} vs {}".format(len(token_type_id), max_seq_len)
        assert len(label) == max_seq_len, "Error with labels length {} vs {}".format(len(label), max_seq_len)

        input_ids.append(input_id)
        attention_masks.append(attention_mask)
        token_type_ids.append(token_type_id)
        data_labels.append(label)

    input_ids = np.array(input_ids, dtype=int)
    attention_masks = np.array(attention_masks, dtype=int)
    token_type_ids = np.array(token_type_ids, dtype=int)
    data_labels = np.asarray(data_labels, dtype=np.int32)

    return (input_ids, attention_masks, token_type_ids), data_labels


# In[17]:


X_train, y_train = convert_examples_to_features(train_data_sentence, train_data_label, max_seq_len=128, tokenizer=tokenizer)


# In[18]:


X_test, y_test = convert_examples_to_features(test_data_sentence, test_data_label, max_seq_len=128, tokenizer=tokenizer)


# In[19]:


from transformers import TFBertForTokenClassification


# In[20]:


# TPU 작동을 위한 코드
resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu='grpc://' + os.environ['COLAB_TPU_ADDR'])
tf.config.experimental_connect_to_cluster(resolver)
tf.tpu.experimental.initialize_tpu_system(resolver)

strategy = tf.distribute.experimental.TPUStrategy(resolver)


# In[21]:


with strategy.scope():
  model = TFBertForTokenClassification.from_pretrained("klue/bert-base", num_labels=tag_size, from_pt=True)
  optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5)
  model.compile(optimizer=optimizer, loss=model.hf_compute_loss)


# In[22]:


class F1score(tf.keras.callbacks.Callback):
    def __init__(self, X_test, y_test):
        self.X_test = X_test
        self.y_test = y_test

    def sequences_to_tags(self, label_ids, pred_ids):
      label_list = []
      pred_list = []

      for i in range(0, len(label_ids)):
        label_tag = []
        pred_tag = []

        for label_index, pred_index in zip(label_ids[i], pred_ids[i]):
          if label_index != -100:
            label_tag.append(index_to_tag[label_index])
            pred_tag.append(index_to_tag[pred_index])
        
        label_list.append(label_tag)
        pred_list.append(pred_tag)

      return label_list, pred_list

    def on_epoch_end(self, epoch, logs={}):
      y_predicted = self.model.predict(self.X_test)
      y_predicted = np.argmax(y_predicted.logits, axis = 2)

      label_list, pred_list = self.sequences_to_tags(self.y_test, y_predicted)

      score = f1_score(label_list, pred_list, suffix=True)
      print(' - f1: {:04.2f}'.format(score * 100))
      print(classification_report(label_list, pred_list, suffix=True))
     


# In[23]:


f1_score_report = F1score(X_test, y_test)


# In[24]:


model.fit(
    X_train, y_train, epochs=3, batch_size=32,
    callbacks = [f1_score_report]
)


# In[25]:


def convert_examples_to_features_for_prediction(examples, max_seq_len, tokenizer,
                                 pad_token_id_for_segment=0, pad_token_id_for_label=-100):
    cls_token = tokenizer.cls_token
    sep_token = tokenizer.sep_token
    pad_token_id = tokenizer.pad_token_id

    input_ids, attention_masks, token_type_ids, label_masks = [], [], [], []

    for example in tqdm(examples):
        tokens = []
        label_mask = []
        for one_word in example:
            subword_tokens = tokenizer.tokenize(one_word)
            tokens.extend(subword_tokens)
            label_mask.extend([0]+ [pad_token_id_for_label] * (len(subword_tokens) - 1))

        special_tokens_count = 2
        if len(tokens) > max_seq_len - special_tokens_count:
            tokens = tokens[:(max_seq_len - special_tokens_count)]
            label_mask = label_mask[:(max_seq_len - special_tokens_count)]

        tokens += [sep_token]
        label_mask += [pad_token_id_for_label]
        tokens = [cls_token] + tokens
        label_mask = [pad_token_id_for_label] + label_mask
        input_id = tokenizer.convert_tokens_to_ids(tokens)
        attention_mask = [1] * len(input_id)
        padding_count = max_seq_len - len(input_id)
        input_id = input_id + ([pad_token_id] * padding_count)
        attention_mask = attention_mask + ([0] * padding_count)
        token_type_id = [pad_token_id_for_segment] * max_seq_len
        label_mask = label_mask + ([pad_token_id_for_label] * padding_count)

        assert len(input_id) == max_seq_len, "Error with input length {} vs {}".format(len(input_id), max_seq_len)
        assert len(attention_mask) == max_seq_len, "Error with attention mask length {} vs {}".format(len(attention_mask), max_seq_len)
        assert len(token_type_id) == max_seq_len, "Error with token type length {} vs {}".format(len(token_type_id), max_seq_len)
        assert len(label_mask) == max_seq_len, "Error with labels length {} vs {}".format(len(label_mask), max_seq_len)

        input_ids.append(input_id)
        attention_masks.append(attention_mask)
        token_type_ids.append(token_type_id)
        label_masks.append(label_mask)

    input_ids = np.array(input_ids, dtype=int)
    attention_masks = np.array(attention_masks, dtype=int)
    token_type_ids = np.array(token_type_ids, dtype=int)
    label_masks = np.asarray(label_masks, dtype=np.int32)

    return (input_ids, attention_masks, token_type_ids), label_masks


# In[26]:


X_pred, label_masks = convert_examples_to_features_for_prediction(test_data_sentence[:5], max_seq_len=128, tokenizer=tokenizer)


# In[27]:


def ner_prediction(examples, max_seq_len, tokenizer):
  examples = [sent.split() for sent in examples]
  X_pred, label_masks = convert_examples_to_features_for_prediction(examples, max_seq_len=128, tokenizer=tokenizer)
  y_predicted = model.predict(X_pred)
  y_predicted = np.argmax(y_predicted.logits, axis = 2)

  pred_list = []
  result_list = []

  for i in range(0, len(label_masks)):
    pred_tag = []
    for label_index, pred_index in zip(label_masks[i], y_predicted[i]):
      if label_index != -100:
        pred_tag.append(index_to_tag[pred_index])

    pred_list.append(pred_tag)

  for example, pred in zip(examples, pred_list):
    one_sample_result = []
    for one_word, label_token in zip(example, pred):
      one_sample_result.append((one_word, label_token))
    result_list.append(one_sample_result)

  return result_list


# In[28]:


sent1 = '오리온스는 리그 최정상급 포인트가드 김동훈을 앞세우는 빠른 공수전환이 돋보이는 팀이다'
sent2 = '하이신사에 속한 섬들도 위로 솟아 있는데 타인은 살고 있어요'


# In[29]:


test_samples = [sent1, sent2]


# In[30]:


result_list = ner_prediction(test_samples, max_seq_len=128, tokenizer=tokenizer)


# In[31]:


result_list

