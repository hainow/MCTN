# get smaller word embedding and word2ix that only contains words in the dataset

import cPickle
import numpy as np


def get_mosi_dictionary():
  path = '../data/mosi/Transcript/Final' \
         '/word2index.txt'
  with open(path) as f:
    lines = f.read().split('\n')
  words = [line.split(' ')[1].lower() for line in lines]
  return set(words)


def create_mosi_we_w2ix():
  words = get_mosi_dictionary()  # lowercase, set
  word2ix = dict()
  filtered_word_embedding = [[0] * 300]
  i = 1
  for w in words:
    if w in we_dict:
      word2ix[w.upper()] = i
      i += 1
      filtered_word_embedding += [we_dict[w]]
    else:
      word2ix[w.upper()] = 0
  filtered_word_embedding = np.array(filtered_word_embedding)
  with open('../data/glove_word_embedding/glove_300_mosi.pkl', 'wb') as f:
    cPickle.dump(filtered_word_embedding, f)
  with open('../data/glove_word_embedding/word2ix_300_mosi.pkl', 'wb') as f:
    cPickle.dump(word2ix, f)


def create_imdb_mosi_we_w2ix():
  path = '../data/aclImdb/imdb.vocab'
  with open(path) as f:
    words_1 = set(f.read().split('\n'))
  words_2 = get_mosi_dictionary()
  words = words_1 | words_2
  word2ix = dict()
  filtered_word_embedding = [[0] * 300]
  i = 1
  for w in words:
    if w in we_dict:
      word2ix[w] = i
      i += 1
      filtered_word_embedding += [we_dict[w]]
    else:
      word2ix[w] = 0

  filtered_word_embedding = np.array(filtered_word_embedding)
  with open('../data/glove_word_embedding/glove_300_imdb-mosi.pkl', 'wb') as f:
    cPickle.dump(filtered_word_embedding, f)

  with open('../data/glove_word_embedding/word2ix_300_imdb-mosi.pkl',
            'wb') as f:
    cPickle.dump(word2ix, f)


data_path = '../data/'
word_embedding_path = data_path + 'glove_word_embedding/glove.840B.300d.txt'
with open(word_embedding_path) as f:
  st = f.read()

lines = st.split('\n')
lines = [line.split(' ') for line in lines if line != '']
we_dict = {line[0]: [float(x) for x in line[1:]] for line in lines}
create_mosi_we_w2ix()
