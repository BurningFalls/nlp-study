#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/BurningFalls/nlp-study/blob/main/BERT%20(SKplanet%20Tacademy)/T_academy_bert.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# # Google Drive 연동 설정

# 실습에 앞서, 모델 파일과 학습 코드가 저장되어 있는 구글 드라이브의 디렉토리와 Colab을 연동하겠습니다.
# 
# 먼저, 좌측 상단 메뉴에서 런타임 -> 런타임 유형 변경 -> 하드웨어 가속기 -> GPU 선택 후 저장해주세요 :-)

# In[ ]:


get_ipython().system('nvidia-smi')


# In[ ]:


from google.colab import drive
drive.mount('/content/drive')


# In[ ]:


import locale
locale.getpreferredencoding = lambda: "UTF-8"


# In[ ]:


get_ipython().system('ls')


# In[ ]:


get_ipython().system('ls drive/My\\ Drive/t_academy')


# # BERT 학습을 위한 vocab을 만들기

# 학습에 사용될 vocab을 만들어보겠습니다.
# 
# 아래 코드를 실행하시면, vocab을 만들 수 있습니다.
# 
# 이번 예제에서는 결과를 빠르게 확인하기 위해 iter를 20으로 두었으나, 
# 
# wiki를 기준으로 1000 정도로 iter를 설정하면 대략 30,000 vocab정도 만들어집니다.

# In[ ]:


get_ipython().system('python drive/My\\ Drive/t_academy/src/make_vocab/wordpiece.py --corpus=drive/My\\ Drive/t_academy/rsc/training_data/wiki_20190620_small.txt --iter=20 --fname=drive/My\\ Drive/t_academy/rsc/conf/vocab.txt')


# #BERT 학습을 위한 Preprocessed data 만들기

# 이제 vocab 이 준비되었으니, BERT 학습을 위한 corpus를 preprocessing 해보도록 하겠습니다.

# In[ ]:


get_ipython().system('python drive/My\\ Drive/t_academy/src/make_preprocessed_data/create_pretraining_data.py --input_file=drive/My\\ Drive/t_academy/rsc/training_data/wiki_20190620_small.txt --vocab_file=drive/My\\ Drive/t_academy/rsc/conf/vocab.txt --do_lower_case=False --max_seq_length=512 --output_file=drive/My\\ Drive/t_academy/rsc/preprocessed_training_data/wiki_20190620_small_512_tf.record')


# # BERT 학습

# 이제 만들어진 학습 데이터를 이용해서 실제로 BERT를 학습해보도록 하겠습니다.
# 
# 이번 학습에선 앞서 만든 wiki_small이 아니라, 제가 미리 만들어 배포드린 전체 wiki 데이터를 이용해 학습해보겠습니다.
# 
# 참고로, train_batch_size를 4보다 크게 할 경우, colab에서 제공하는 GPU로는 메모리가 부족해서 에러가 발생합니다.

# In[ ]:


get_ipython().system('python drive/My\\ Drive/t_academy/src/make_bert_model/run_pretraining.py --input_file=drive/My\\ Drive/t_academy/rsc/preprocessed_training_data/wiki_20190620_512_tf.record --output_dir=drive/My\\ Drive/t_academy/rsc/pretrained_model --do_train=True --do_eval=True --bert_config_file=drive/My\\ Drive/t_academy/rsc/conf/bert_config.json --train_batch_size=4 --max_seq_length=512 --max_predictions_per_seq=20 --num_train_steps=10 --learning_rate=1e-4 --save_checkpoints_steps=5 --do_lower_case=False')


# # 학습 된 BERT 모델로 KorQuAD 학습

# 이번엔 사전에 제공해드린 BERT 모델을 이용해 KorQuAD를 학습해보도록 하겠습니다.
# 
# max_seq_length와 num_train_epochs 등을 줄여서 학습 시간을 줄였지만, 
# 
# 그래도 대략 30분 정도 소요됩니다 :-)
# 
# Default parameter로 학습을 하면 대략 4시간 정도 학습이 소요됩니다.

# In[ ]:


get_ipython().system('python drive/My\\ Drive/bert_for_practics/t_academy/src/make_bert_model/run_squad.py --vocab_file=drive/My\\ Drive/bert_for_practics/t_academy/rsc/conf/vocab.txt --bert_config_file=drive/My\\ Drive/bert_for_practics/t_academy/rsc/conf/bert_config.json --init_checkpoint=drive/My\\ Drive/bert_for_practics/t_academy/rsc/pretrained_model/model_output_512_model.ckpt-200000 --do_train=True --train_file=drive/My\\ Drive/bert_for_practics/t_academy/rsc/KorQuAD/KorQuAD_v1.0_train.json --do_predict=True --predict_file=drive/My\\ Drive/bert_for_practics/t_academy/rsc/KorQuAD/KorQuAD_v1.0_dev.json --train_batch_size=16 --learning_rate=2e-5 --num_train_epochs=1.0 --max_seq_length=128 --doc_stride=128 --output_dir=drive/My\\ Drive/bert_for_practics/t_academy/rsc/KorQuAD_output --do_lower_case=False')


# 이제 구글 드라이브에 predictions.json 이라는 이름으로 KorQuAD dev set에 대한 prediction 결과가 저장됐습니다.

# # 학습 된 KorQuAD 평가

# 아래 코드를 실행해보시면, KorQuAD를 평가하실 수 있습니다.
# 
# 결과가 실망스러우신가요? :-)
# 
# Defualt parameter로 KorQuAD 학습을 해보시면,
# 
# {"exact_match": 67.23242119847593, "f1": 86.38759335665746}
# 
# 정도의 결과를 확인하실 수 있습니다 :-)
# 
# 역시 높은 수치는 아니지만, wiki 데이터만을 이용해 20만 step만 학습한 BERT 모델로는
# 
# 괜찮게 나온 결과라고 생각합니다 :-)
# 
# ---
# 
# 

# In[ ]:


get_ipython().system('python drive/My\\ Drive/bert_for_practics/t_academy/rsc/KorQuAD/evaluate-v1.0.py drive/My\\ Drive/bert_for_practics/t_academy/rsc/KorQuAD/KorQuAD_v1.0_dev.json drive/My\\ Drive/bert_for_practics/t_academy/rsc/KorQuAD_output/predictions.json')


# # BERT 감정 데이터 분류 실습

# 이번에는 네이버 영화 리뷰 데이터와 BERT를 이용해 문장 분류를 실습해보겠습니다.
# 
# 네이버 영화 리뷰 데이터는 아래의 주소에서 다운받을 수 있습니다.
# 
# https://github.com/e9t/nsmc
# 
# BERT에서는 run classification 코드를 통해 분류 학습을 수행할 수 있습니다.
# 
# 약 20분 정도 소요됩니다 :-)

# In[ ]:


get_ipython().system('python drive/My\\ Drive/bert_for_practics/t_academy/src/make_bert_model/run_classifier.py --task_name=nsmc --do_train=true --do_eval=true --data_dir=drive/My\\ Drive/bert_for_practics/t_academy/rsc/nsmc --vocab_file=drive/My\\ Drive/bert_for_practics/t_academy/rsc/conf/vocab.txt --bert_config_file=drive/My\\ Drive/bert_for_practics/t_academy/rsc/conf/bert_config.json --init_checkpoint=drive/My\\ Drive/bert_for_practics/t_academy/rsc/pretrained_model/model_output_512_model.ckpt-200000 --max_seq_length=128 --train_batch_size=32 --num_train_epochs=1.0 --learning_rate=3e-5 --do_lower_case=false --output_dir=drive/My\\ Drive/bert_for_practics/t_academy/rsc/nsmc_output')


# 정확도가 86%정도 나왔네요!
# 
# 짧게 학습을 수행한 것 치고는 매우 높게 잘 나온 것 같습니다 :-)

# # BERT 관계 추출 실습

# 이번 실습에서는 BERT를 이용해 entity가 가지는 관계를 추출해보는 실습을 해보도록 하겠습니다.
# 
# 관계 추출 데이터는 Kaist가 공개한 데이터를 사용하였습니다.
# 
# 해당 데이터는 다음의 사이트에서 다운로드 받으실 수 있습니다.
# 
# https://github.com/machinereading/kor-re-gold
# 
# 본 테스트를 위해 학습 데이터 n건과 테스트 데이터 n건으로 나눠서 전처리를 수행해두었습니다.
# 
# 기존 BERT와 달리, 이 경우엔 다음과 같이 문장이 입력됩니다.
# 
# Subject[SEP]Object[SEP]Sentence

# In[ ]:


get_ipython().system('python drive/My\\ Drive/bert_for_practics/t_academy/src/make_bert_model/run_multi_classifier.py --task_name=kent --do_train=true --do_eval=true --data_dir=drive/My\\ Drive/bert_for_practics/t_academy/rsc/relation --vocab_file=drive/My\\ Drive/bert_for_practics/t_academy/rsc/conf/vocab.txt --bert_config_file=drive/My\\ Drive/bert_for_practics/t_academy/rsc/conf/bert_config.json --init_checkpoint=drive/My\\ Drive/bert_for_practics/t_academy/rsc/pretrained_model/model_output_512_model.ckpt-200000 --max_seq_length=128 --train_batch_size=32 --num_train_epochs=1.0 --learning_rate=2e-5 --do_lower_case=false --output_dir=drive/My\\ Drive/bert_for_practics/t_academy/rsc/relation_output')


# 학습 시간이 정말 오래걸리죠? :-)
# 
# 발표 장표에 첨부된 85%의 결과는 거의 하루 가까이 돌렸던 것 같네요 :-)

# In[ ]:




