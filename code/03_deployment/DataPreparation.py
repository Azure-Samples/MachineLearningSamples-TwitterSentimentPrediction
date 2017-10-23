import numpy as np
#import cPickle as pickle
import pickle


class Data_Preparation:

    def __init__ (self, vectors_file):
        self.vectors_dc=self.load_word_embedding_pickle(vectors_file)

    
    def load_word_embedding_pickle(self, pickle_file):
        vectors_dc = pickle.load( open( pickle_file, "rb" ) )
        return vectors_dc
    
    def get_sentence_embedding(self,text_data ):
        vectors_dc=self.vectors_dc
        sentence_vectors=[]

        for sen in text_data:
            tokens=sen.split(' ')
            current_vector=np.array([vectors_dc[tokens[0].encode('utf-8')] if tokens[0].encode('utf-8') in vectors_dc else vectors_dc[b'<UNK>']])
            for word in tokens[1:]:
                if word.encode('utf-8') in vectors_dc:
                    current_vector=np.vstack([current_vector,vectors_dc[word.encode('utf-8')]])
                else:
                    current_vector=np.vstack([current_vector,vectors_dc[b'<UNK>']])
            min_max_mean=np.hstack([current_vector.min(axis=0),current_vector.max(axis=0),current_vector.mean(axis=0)])
            sentence_vectors.append(min_max_mean)
        return sentence_vectors