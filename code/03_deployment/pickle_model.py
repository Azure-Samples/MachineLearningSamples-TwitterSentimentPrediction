import os
from os import path
import json
import pip
import pickle as pickle
import numpy as np

def load_word_embedding( vectors_file):
    vectors= np.genfromtxt(vectors_file, delimiter='\t', comments='#--#',dtype=None, 
                           names=['Word']+['EV{}'.format(i) for i in range(1,51)])
    vectors_dc={}
    for x in vectors:
        vectors_dc[x['Word']]=[x[each] for each in ['EV{}'.format(i) for i in range(1,51)]]
    return vectors_dc

def picklify(raw_embedding_file, dest_dir='./'):
    dc=load_word_embedding(raw_embedding_file)
    pickle.dump(dc,open(dest_dir+'//pickle_'+raw_embedding_file,'wb'))

raw_embedding_file1 = 'embeddings_Word2Vec_Basic.tsv'
picklify(raw_embedding_file1)

raw_embedding_file2 = 'embeddings_SSWE_Basic_Keras_w_CNTK.tsv'
picklify(raw_embedding_file2)