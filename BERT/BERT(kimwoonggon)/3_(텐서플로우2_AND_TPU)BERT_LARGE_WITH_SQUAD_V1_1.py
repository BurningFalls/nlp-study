#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/BurningFalls/nlp-study/blob/main/BERT%20(%EA%B9%80%EC%9B%85%EA%B3%A4)/3_(%ED%85%90%EC%84%9C%ED%94%8C%EB%A1%9C%EC%9A%B02_AND_TPU)BERT_LARGE_WITH_SQUAD_V1_1.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# In[ ]:


get_ipython().system('pip install transformers')


# In[ ]:


get_ipython().run_line_magic('tensorflow_version', '2.x')
import tensorflow as tf
import numpy as np
import pandas as pd
from transformers import *
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
import os


# In[ ]:


import os
from google.colab import drive
drive.mount('/content/gdrive/')


# In[ ]:


path = "gdrive/My Drive/Colab Notebooks/squad"


# In[ ]:


tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')


# In[ ]:


get_ipython().system('wget https://raw.githubusercontent.com/nate-parrott/squad/master/data/train-v1.1.json')
get_ipython().system('wget https://raw.githubusercontent.com/nate-parrott/squad/master/data/dev-v1.1.json')


# In[ ]:


def squad_json_to_dataframe_train(input_file_path, record_path = ['data','paragraphs','qas','answers'],
                           verbose = 1):
    """
    input_file_path: path to the squad json file.
    record_path: path to deepest level in json file default value is
    ['data','paragraphs','qas','answers']
    verbose: 0 to suppress it default is 1
    """
    if verbose:
        print("Reading the json file")    
    file = json.loads(open(input_file_path).read())
    if verbose:
        print("processing...")
    # parsing different level's in the json file
    js = pd.io.json.json_normalize(file , record_path )
    m = pd.io.json.json_normalize(file, record_path[:-1] )
    r = pd.io.json.json_normalize(file,record_path[:-2])
    
    #combining it into single dataframe
    idx = np.repeat(r['context'].values, r.qas.str.len())
    ndx  = np.repeat(m['id'].values,m['answers'].str.len())
    m['context'] = idx
    js['q_idx'] = ndx
    main = pd.concat([ m[['id','question','context']].set_index('id'),js.set_index('q_idx')],1,sort=False).reset_index()
    main['c_id'] = main['context'].factorize()[0]
    if verbose:
        print("shape of the dataframe is {}".format(main.shape))
        print("Done")
    return main


# In[ ]:


train = squad_json_to_dataframe_train("train-v1.1.json")
# context의 길이를 알려주는 칼럼 생성
train['context_len'] = train['context'].str.len()
# 질문의 길이가 10 미만인 데이터 확인
# 질문의 길이가 10 미만이면 이상한 데이터일 가능성 높음
train.loc[train['question'].str.len() <= 10].head(10)
# 질문의 길이가 10 미만인 데이터 삭제
train = train.loc[train['question'].str.len() >= 10].reset_index(drop=True)


# In[ ]:


def convert_data(data_df):
    global tokenizer
    indices, segments, masks, target_start, target_end = [], [], [], [], []
    
    for i in tqdm(range(len(data_df))):
        # que : question을 버트 인풋으로 들어갈 수 있게 토큰화 한 변수, tokenizer.encode를 사용하면
        # 토큰화 된 리스트와, 세그먼트가 같이 나오는데 일단 토큰화 된 리스트만 사용

        # doc : context를 버트 인풋으로 들어갈 수 있게 토근화 한 변수
        que = tokenizer.encode(data_df[QUESTION_COLUMN][i])
        doc = tokenizer.encode(data_df[DATA_COLUMN][i])
        
        # 토큰화된 context의 맨 앞에 있는 [CLS]에 해당하는 101을 삭제
        doc.pop(0)

        # que_len, doc_len : 질문의 길이, context의 길이
        que_len = len(que)
        doc_len = len(doc)

        # 만약 question의 길이가 64를 초과하면, 64로 잘라줌

        if que_len > 64:
          que = que[:63]
          # 질문의 끝이 [SEP]이 되도록, [SEP]에 해당하는 3 추가
          que.append(102)
        
        # 버트 인풋으로 들어가는 토큰화된 리스트가 최대 길이인 384가 넘지 않도록 만들어 줌
        # 384 미만이면 context를 잘라줌
        if len(que+doc) > SEQ_LEN:
          while len(que+doc) != SEQ_LEN:
            doc.pop(-1)
          doc.pop(-1)
          #context의 끝이 [SEP]가 되도록 [SEP]에 해당하는 102를 추가해 줌
          doc.append(102)

        # 문장의 전후관계를 구분해주는 segment는, question은 0이 되도록, context는 1이 되도록, 나머지 부분인 패딩 부분은
        # 0이 되도록 만들어 줌
        
        ############################
        ###### Segment 예시 ########
        ############################
        
        # question, context, padding
        # 00000000, 1111111, 0000000
        
        segment = [0]*len(que) + [1]*len(doc) + [0]*(SEQ_LEN-len(que)-len(doc))
        if len(que + doc) <= SEQ_LEN:
          mask = [1]*len(que+doc) + [0]*(SEQ_LEN-len(que+doc))
        else:
          mask = [1]*len(que+doc)
        # 만약 question과 context를 합쳤을 때 그 길이가 384 미만이면
        # padding 값인 0을 채워주도록 함
        if len(que + doc) <= SEQ_LEN:
          while len(que+doc) != SEQ_LEN:
            doc.append(0)

        # ids : question과 context를 합친 버트의 실질적인 인풋

        ids = que + doc
        
        # text 길이만큼 context를 sliding 하면서, context 안에 일치하는 text를 찾았을 경우
        # context 내에 text의 시작 위치와 끝 위치를 알려주는 부분 코딩
        
        text = tokenizer.encode(data_df[TEXT][i])
        text_slide_len = len(text[1:-1])
        
        # exist_flag : context 내에서 text를 찾았을 경우 0에서 1로 전환
        for j in range(0,(len(doc))):  
            exist_flag = 0
            if text[1:-1] == doc[j:j+text_slide_len]:
              ans_start = j + len(que)
              ans_end = j + text_slide_len - 1 + len(que)
              exist_flag = 1
              break
        
        # 만약 context 내에서 text를 찾지 못해서 여전히 exist_flag 가 0인 경우
        # 시작값과 끝 값을 SEQ_LEN(384로 지정)
        # 향후 시작값과 끝 값이 384인 경우 이 목록은 삭제할 예정임
        if exist_flag == 0:
          ans_start = SEQ_LEN
          ans_end = SEQ_LEN

        # 버트 인풋으로 들어가는 ids, segments를 indices, segments에 각각 저장
        indices.append(ids)
        segments.append(segment)
        masks.append(mask)
        # 정답에 해당하는 시작 위치인 ans_start와 ans_end를 target_start, target_end에 각각 저장
        target_start.append(ans_start)
        target_end.append(ans_end)

    # indices, segments, ans_start, ans_end를 numpy array로 지정    
    indices_x = np.array(indices)
    segments = np.array(segments)
    masks = np.array(masks)
    target_start = np.array(target_start)
    target_end = np.array(target_end)
    
    # del_list를 지정하여 ans_start와 ans_end가 정답에 해당하지 않는 부분들을 삭제
    del_list = np.where(target_start!=SEQ_LEN)[0]
    not_del_list = np.where(target_start==SEQ_LEN)[0]
    indices_x = indices_x[del_list]
    segments = segments[del_list]
    masks = masks[del_list]
    target_start = target_start[del_list]
    target_end = target_end[del_list]

    return [indices_x, masks, segments], [target_start, target_end], not_del_list

# 위에 정의한 convert_data 함수를 불러오는 함수를 정의
def load_data(pandas_dataframe):
    data_df = pandas_dataframe
    data_df[DATA_COLUMN] = data_df[DATA_COLUMN].astype(str)
    data_df[QUESTION_COLUMN] = data_df[QUESTION_COLUMN].astype(str)
    data_df[TEXT] = data_df[TEXT].astype(str)
    data_x, data_y, del_list = convert_data(data_df)

    return data_x, data_y, del_list

SEQ_LEN = 384
DATA_COLUMN = "context"
# context를 포함하고 있는 열의 이름
QUESTION_COLUMN = "question"
# question을 포함하고 있는 열의 이름
TEXT = "text"
# text(정답)을 포함하고 있는 열의 이름

train_x, train_y, z = load_data(train)


# In[ ]:


resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu='grpc://' + os.environ['COLAB_TPU_ADDR'])
tf.config.experimental_connect_to_cluster(resolver)
tf.tpu.experimental.initialize_tpu_system(resolver)


# In[ ]:


# 깔끔한 모델
def create_model2():
  
  model = TFBertModel.from_pretrained('bert-large-uncased')
  token_inputs = tf.keras.layers.Input((SEQ_LEN,), dtype=tf.int32, name='input_word_ids')
  seg_inputs = tf.keras.layers.Input((SEQ_LEN,), dtype=tf.int32, name='input_segments')
  mask_inputs = tf.keras.layers.Input((SEQ_LEN,), dtype=tf.int32, name='input_masks')

  seq_output, _ = model([token_inputs, mask_inputs, seg_inputs])
  x = tf.keras.layers.Dense(2, kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02))(seq_output)
  start, end = tf.split(x, 2, axis=-1)
  start = tf.squeeze(start, axis=-1)
  end = tf.squeeze(end, axis=-1)
  bert_model2 = tf.keras.Model([token_inputs, mask_inputs, seg_inputs], [start, end])
  import tensorflow_addons as tfa
  #opt = tfa.optimizers.RectifiedAdam(lr=5e-5, warmup_proportion=0.1, total_steps=10000)
  opt = tf.keras.optimizers.Adam(lr=1.5e-5)
  bert_model2.compile(
      optimizer = opt,
      loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
      metrics=['sparse_categorical_accuracy'])
  bert_model2.summary()
  del model
  return bert_model2


# In[ ]:


strategy = tf.distribute.experimental.TPUStrategy(resolver)
with strategy.scope(): # creating the model in the TPUStrategy scope means we will train the model on the TPU
  bert_model2 = create_model2()
  
bert_model2.fit(train_x, train_y, epochs=2, shuffle=True, batch_size=18)


# In[ ]:


bert_model2.save_weights(os.path.join(path, "bert_large_2epoch.h5"))


# In[ ]:


# 2 EPOCH
from sklearn.metrics import classification_report
preds = bert_model2.predict(train_x)

start_indexes = np.argmax(preds[0], axis=-1)
end_indexes = np.argmax(preds[1], axis=-1)

# start_index의 f1_score
print(classification_report(train_y[0], start_indexes))

# end_index의 f1_score
print(classification_report(train_y[1], end_indexes))


# In[ ]:


def squad_json_to_dataframe_dev(input_file_path, record_path = ['data','paragraphs','qas','answers'],
                           verbose = 1):
    """
    input_file_path: path to the squad json file.
    record_path: path to deepest level in json file default value is
    ['data','paragraphs','qas','answers']
    verbose: 0 to suppress it default is 1
    """
    if verbose:
        print("Reading the json file")    
    file = json.loads(open(input_file_path).read())
    if verbose:
        print("processing...")
    # parsing different level's in the json file
    js = pd.io.json.json_normalize(file , record_path )
    m = pd.io.json.json_normalize(file, record_path[:-1] )
    r = pd.io.json.json_normalize(file,record_path[:-2])
    
    #combining it into single dataframe
    idx = np.repeat(r['context'].values, r.qas.str.len())
    m['context'] = idx
    main = m[['id','question','context','answers']].set_index('id').reset_index()
    main['c_id'] = main['context'].factorize()[0]
    if verbose:
        print("shape of the dataframe is {}".format(main.shape))
        print("Done")
    return main


# In[ ]:


input_file_path ='dev-v1.1.json'
record_path = ['data','paragraphs','qas','answers']
verbose = 0
dev = squad_json_to_dataframe_dev(input_file_path=input_file_path,record_path=record_path)


# In[ ]:


# 정답의 개수를 정의하는 칼럼 생성
dev['answer_len'] = dev['answers'].map(lambda x: len(x))


# In[ ]:


# 정답들을 다루기 쉽게 리스트로 반환하는 함수 정의
def get_text(text_len, answers):
  # text_len : 질문(question)과 문장(context)에 해당하는 정답의 개수
  # answers : 정답 ex) [{'answer_start': 177, 'text': 'Denver Broncos'}, {'answer_start': 177, 'text': 'Denver Broncos'}, {'answer_start': 177, 'text': 'Denver Broncos'}]
  texts = []
  for i in range(text_len):
    texts.append(answers[i]['text'])
  return texts


# In[ ]:


# texts 칼럼의 모든 데이터에 대해서 수행
dev['texts'] = dev.apply(lambda x: get_text(x['answer_len'], x['answers']), axis=1)


# In[ ]:


TEXT_COLUMN = 'texts'


# In[ ]:


def convert_data(data_df):
    global tokenizer
    indices, segments, masks, target_start, target_end = [], [], [], [], []

    for i in tqdm(range(len(data_df))):
        que = tokenizer.encode(data_df[QUESTION_COLUMN][i])
        doc = tokenizer.encode(data_df[DATA_COLUMN][i])
        doc.pop(0)

        que_len = len(que)
        doc_len = len(doc)

        if que_len > 64:
          que = que[:63]
          que.append(102)
        
        if len(que+doc) > SEQ_LEN:
          while len(que+doc) != SEQ_LEN:
            doc.pop(-1)

          doc.pop(-1)
          doc.append(102)
        
        if len(que + doc) <= SEQ_LEN:
          mask = [1]*len(que+doc) + [0]*(SEQ_LEN-len(que+doc))
        else:
          mask = [1]*len(que+doc)
        segment = [0]*len(que) + [1]*len(doc) + [0]*(SEQ_LEN-len(que)-len(doc))
        if len(que + doc) <= SEQ_LEN:
          while len(que+doc) != SEQ_LEN:
            doc.append(0)

        ids = que + doc

        texts = data_df[TEXT_COLUMN][i]
        for text_element in texts:
          text = tokenizer.encode(text_element)

          text_slide_len = len(text[1:-1])
          for j in range(0,(len(doc))):  
              exist_flag = 0
              if text[1:-1] == doc[j:j+text_slide_len]:
                ans_start = j + len(que)
                ans_end = j + text_slide_len - 1 + len(que)
                exist_flag = 1
                break
        
          if exist_flag == 0:
            ans_start = SEQ_LEN
            ans_end = SEQ_LEN

        indices.append(ids)
        segments.append(segment)
        masks.append(mask)
        target_start.append(ans_start)
        target_end.append(ans_end)
        


    # indices, segments, ans_start, ans_end를 numpy array로 지정    
    indices_x = np.array(indices)
    segments = np.array(segments)
    masks = np.array(masks)
    target_start = np.array(target_start)
    target_end = np.array(target_end)
    
    # del_list를 지정하여 ans_start와 ans_end가 정답에 해당하지 않는 부분들을 삭제
    del_list = np.where(target_start!=SEQ_LEN)[0]
    not_del_list = np.where(target_start==SEQ_LEN)[0]
    indices_x = indices_x[del_list]
    segments = segments[del_list]
    masks = masks[del_list]
    target_start = target_start[del_list]
    target_end = target_end[del_list]

    return [indices_x, masks, segments], del_list

def load_data(pandas_dataframe):
    data_df = pandas_dataframe
    data_df[DATA_COLUMN] = data_df[DATA_COLUMN].astype(str)
    data_df[QUESTION_COLUMN] = data_df[QUESTION_COLUMN].astype(str)
    data_x, data_y, del_list = convert_data(data_df)

    return data_x, del_list


# In[ ]:


dev_bert_input = convert_data(dev)


# In[ ]:



dev_bert_input, del_list = dev_bert_input[0], dev_bert_input[1]
dev = dev.iloc[del_list]
dev = dev.reset_index(drop=True)


# In[ ]:


indexes = dev_bert_input[0]
bert_predictions = bert_model2.predict(dev_bert_input)


# In[ ]:


start_indexes = np.argmax(bert_predictions[0], axis=-1)
end_indexes = np.argmax(bert_predictions[1], axis=-1)
not_del_list = np.where(start_indexes <= end_indexes)[0]
start_indexes = start_indexes[not_del_list]
end_indexes = end_indexes[not_del_list]
indexes = indexes[not_del_list]


# In[ ]:


dev = dev.iloc[not_del_list].reset_index(drop=True)


# In[ ]:


tokenizer.convert_ids_to_tokens(102)


# In[ ]:


# length : dev 데이터의 길이
length = len(dev)

sentences = []

untokenized = []

for j in range(len(start_indexes)):
  sentence = []
  for i in range(start_indexes[j], end_indexes[j]+1):
    token_based_word = tokenizer.convert_ids_to_tokens(indexes[j][i].item())
    sentence.append(token_based_word)
    # 문장이 토큰화된 단어 하나 하나를 sentence에 저장
  
  sentence_string = ""
  
  for w in sentence:
    
    if w.startswith("##"):
      w = w.replace("##", "")
      # 만약 sentence 안의 토큰이 ##으로 시작한다면, ##을 제거
    else:
      w = " " + w
      # 토큰이 ##으로 시작하지 않는다면 글자의 첫 시작이므로, 띄어쓰기 추가
    sentence_string += w
      # 리스트로 되어 있는 토큰들을 하나로 합쳐줌
  if sentence_string.startswith(" "):
    sentence_string = "" + sentence_string[1:]
    # sentence_string이 " "로 시작하는 경우에는 띄어쓰기를 삭제
  untokenized.append(sentence_string)
  # 리스트로 되어있는 토큰들을 하나로 합쳐준 것, 이것을 untokenized에 저장
  sentences.append(sentence)


# In[ ]:



dev_answers = []
for i in range(length):
  dev_answer = []
  texts_dict = dev['answers'][i]
  
  for j in range(len(texts_dict)):
    dev_answer.append(texts_dict[j]['text'])
    # 정답 하나 하나를 리스트로 저장
  dev_answers.append(dev_answer)


# In[ ]:


dev_tokens = []
for i in dev_answers:
  dev_tokened = []
  for j in i:
    temp_token = tokenizer.tokenize(j)
    #print(temp_token)
    # 정답을 토큰화
    #temp_token.pop(0)
    # [CLS] 제거
    #temp_token.pop(-1)
    # [SEP] 제거
    dev_tokened.append(temp_token)
  dev_tokens.append(dev_tokened)


# In[ ]:


# 토큰화된 정답을 문장으로 변환시켜주고 합쳐줌
dev_answer_lists = []
for dev_answers in dev_tokens:
  dev_answer_list = []
  for dev_answer in dev_answers:
    dev_answer_string = " ".join(dev_answer)
    dev_answer_list.append(dev_answer_string)
  dev_answer_lists.append(dev_answer_list)


# In[ ]:


# untokenizing
dev_strings_end = []
for dev_strings in dev_answer_lists:
  dev_strings_processed = []
  for dev_string in dev_strings:
    dev_string = dev_string.replace(" ##", "")
    dev_strings_processed.append(dev_string)
  dev_strings_end.append(dev_strings_processed)


# In[ ]:


dev_answers = dev_strings_end


# In[ ]:


from collections import Counter
import string, re

def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def exact_match_score(prediction, ground_truth):
    return (normalize_answer(prediction) == normalize_answer(ground_truth))

def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)


# In[ ]:


f1_sum = 0

for i in range(len(untokenized)):
  f1 = metric_max_over_ground_truths(f1_score, untokenized[i], dev_answers[i])
  f1_sum += f1
print("f1 score : ", f1_sum / length)


# In[ ]:


EM_sum = 0

for i in range(len(untokenized)):
  
  EM = metric_max_over_ground_truths(exact_match_score, untokenized[i], dev_answers[i])
  EM_sum += EM
print("EM Score : ", EM_sum / length)

