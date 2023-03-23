#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/BurningFalls/nlp-study/blob/main/Byte%20Pair%20Encoding/BPE.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# In[124]:


def print_list(my_list):
  for word in my_list:
    for ch in word:
      print(ch, end=' ')
    print()

def print_bilist(my_bilist):
  for word in my_bilist:
    for i in range(len(word)):
      print(f'({word[i][0]}, {word[i][1]})', end = ' ')
    print()

def print_dict(my_dict):
  idx = 0
  for key, value in my_dict.items():
    print(f'{key}: {value}', end='   ')
    idx+=1
    if idx % 3 == 0:
      print()


# In[125]:


sentence = "경찰청 철창살은 외철창살이고 검찰청 철창살은 쌍철창살이다"


# In[126]:


word_list = sentence.split()
print(*word_list, sep='\n')


# In[127]:


for idx, word in enumerate(word_list):
  ch_list = list()
  for idx2, ch in enumerate(word):
    if idx2 != 0:
      ch = "##" + ch
    ch_list.append(ch)
  word_list[idx] = ch_list

print_list(word_list)


# 여기로 다시 돌아와서 iteration 수행

# In[144]:


vocab = set()
for word in word_list:
  for ch in word:
    vocab.add(ch)

print(f'vocab = {vocab}')


# In[140]:


bigram_list = list()

for word in word_list:
  tmp_list = list()
  for i in range(len(word) - 1):
    tmp_list.append([word[i], word[i+1]])
  bigram_list.append(tmp_list)

print_bilist(bigram_list)


# In[141]:


freq = dict()
for bigram in bigram_list:
  for ch in bigram:
    tup = (ch[0], ch[1])
    if tup in freq:
      freq[tup] += 1
    elif tup not in freq:
      freq[tup] = 1

print_dict(freq)


# In[142]:


max_key = ''
maxi = 0
for key, value in freq.items():
  if maxi < value:
    maxi = value
    max_key = key

print(f'{max_key}: {maxi}')


# In[143]:


def Concat_Word(word1, word2):
  ans = ""
  if word1[0] == '#':
    ans += "##"
    for i in range(2, len(word1)):
      ans += word1[i]
  else:
    ans += word1
  if word2[0] == '#':
    for i in range(2, len(word2)):
      ans += word2[i]
  else:
    ans += word2
  return ans
  

for pos, word in enumerate(word_list):
  tmp_list = list()
  idx = 0
  while(idx < len(word)):
    if word[idx] == max_key[0] and word[idx + 1] == max_key[1]:
      tmp_list.append(Concat_Word(word[idx], word[idx + 1]))
      idx += 2
    else:
      tmp_list.append(word[idx])
      idx += 1
  word_list[pos] = tmp_list

print_list(word_list)

