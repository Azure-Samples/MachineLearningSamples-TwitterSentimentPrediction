# The purpose of this notebook is to train a gradient boosting based model to classify the tweets' sentiment as positive or negative.

import numpy as np
import pandas as pd
import os
import io

random_seed=123
np.random.seed(random_seed)

import tensorflow as tf
import keras
from keras import backend as K
from keras.models import Model
from keras.layers import Input, merge
from keras.layers.core import Lambda
from keras import optimizers
from keras import regularizers
from keras.models import load_model
from keras.callbacks import ModelCheckpoint
from keras.utils.np_utils import to_categorical
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Input, Dense, Flatten, Embedding , Activation
from nltk.tokenize import TweetTokenizer
import re
import num2words
from timeit import default_timer as timer
from sklearn import  metrics
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import KFold
from sklearn.externals import joblib


from azureml.logging import get_azureml_logger
from azureml.dataprep.package import run
# initialize the logger
run_logger = get_azureml_logger() 

# change num_estimators and you will likely get a different accuracy.
num_estimators = 5
# load num_estimators from argument if present
if len(sys.argv) > 1:
    num_estimators = int(sys.argv[1])
    if num_estimators < 1:
        num_estimators = 1
run_logger.log("Num Estimators", num_estimators)


# Path of the training file'
data_dir = r'C:\Users\ds1\Documents\AzureML\data'  

# Path of the word vectors
vectors_file = r'C:\Users\ds1\Documents\AzureML\Twitter_Sentiment_NLP_1012\code\02_modeling\vectors\embeddings_Word2Vec_Basic.tsv'
model_identifier = 'evaluation_word2vec_gbm'
models_dir = 'model'

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

vector_size=50

# Read and preprocess the data

pos_emoticons=["(^.^)","(^-^)","(^_^)","(^_~)","(^3^)","(^o^)","(~_^)","*)",":)",":*",":-*",":]",":^)",":}",
               ":>",":3",":b",":-b",":c)",":D",":-D",":O",":-O",":o)",":p",":-p",":P",":-P",":Þ",":-Þ",":X",
               ":-X",";)",";-)",";]",";D","^)","^.~","_)m"," ~.^","<=8","<3","<333","=)","=///=","=]","=^_^=",
               "=<_<=","=>.<="," =>.>="," =3","=D","=p","0-0","0w0","8D","8O","B)","C:","d'-'","d(>w<)b",":-)",
               "d^_^b","qB-)","X3","xD","XD","XP","ʘ‿ʘ","❤","💜","💚","💕","💙","💛","💓","💝","💖","💞",
               "💘","💗","😗","😘","😙","😚","😻","😀","😁","😃","☺","😄","😆","😇","😉","😊","😋","😍",
               "😎","😏","😛","😜","😝","😮","😸","😹","😺","😻","😼","👍"]

neg_emoticons=["--!--","(,_,)","(-.-)","(._.)","(;.;)9","(>.<)","(>_<)","(>_>)","(¬_¬)","(X_X)",":&",":(",":'(",
               ":-(",":-/",":-@[1]",":[",":\\",":{",":<",":-9",":c",":S",";(",";*(",";_;","^>_>^","^o)","_|_",
               "`_´","</3","<=3","=/","=\\",">:(",">:-(","💔","☹️","😌","😒","😓","😔","😕","😖","😞","😟",
               "😠","😡","😢","😣","😤","😥","😦","😧","😨","😩","😪","😫","😬","😭","😯","😰","😱","😲",
               "😳","😴","😷","😾","😿","🙀","💀","👎"]

# Emails
emailsRegex=re.compile(r'[\w\.-]+@[\w\.-]+')

# Mentions
userMentionsRegex=re.compile(r'(?<=^|(?<=[^a-zA-Z0-9-_\.]))@([A-Za-z]+[A-Za-z0-9]+)')

#Urls
urlsRegex=re.compile('r(f|ht)(tp)(s?)(://)(.*)[.|/][^ ]+') # It may not be handling all the cases like t.co without http

#Numerics
numsRegex=re.compile(r"\b\d+\b")

punctuationNotEmoticonsRegex=re.compile(r'(?<=\w)[^\s\w](?![^\s\w])')

emoticonsDict = {} # define desired replacements here
for i,each in enumerate(pos_emoticons):
    emoticonsDict[each]=' POS_EMOTICON_'+num2words.num2words(i).upper()+' '
    
for i,each in enumerate(neg_emoticons):
    emoticonsDict[each]=' NEG_EMOTICON_'+num2words.num2words(i).upper()+' '
    
# use these three lines to do the replacement
rep = dict((re.escape(k), v) for k, v in emoticonsDict.items())
emoticonsPattern = re.compile("|".join(rep.keys()))
 
def read_data(filename):
    """Read the raw tweet data from a file. Replace Emails etc with special tokens"""
    with open(filename, 'r') as f:
    
        all_lines=f.readlines()
        padded_lines=[]
        for line in all_lines:
                    line = emoticonsPattern.sub(lambda m: rep[re.escape(m.group(0))], line.lower().strip())
                    line = userMentionsRegex.sub(' USER ', line )
                    line = emailsRegex.sub(' EMAIL ', line )
                    line=urlsRegex.sub(' URL ', line)
                    line=numsRegex.sub(' NUM ',line)
                    line=punctuationNotEmoticonsRegex.sub(' PUN ',line)
                    line=re.sub(r'(.)\1{2,}', r'\1\1',line)
                    words_tokens=[token for token in TweetTokenizer().tokenize(line)]                   
                    line= ' '.join(token for token in words_tokens )        
                    padded_lines.append(line)
        return padded_lines
    
def read_labels(filename):
    """ read the tweet labels from the file"""
    arr= np.genfromtxt(filename, delimiter='\n')
    arr[arr==4]=1 # Encode the positive category as 1
    return arr


# Convert Word Vectors to Sentence Vectors

def load_word_embedding(vectors_file):
    """ Load the word vectors"""
    vectors= np.genfromtxt(vectors_file, delimiter='\t', comments='#--#',dtype=None,
                           names=['Word']+['EV{}'.format(i) for i in range(1,51)]) #51 is embedding length + 1, change accoridngly if the size of embedding is not 50
    vectors_dc={}
    for x in vectors:
        vectors_dc[x['Word'].decode('utf-8','ignore')]=[float(x[each]) for each in ['EV{}'.format(i) for i in range(1,51)]] #51 is embedding length + 1, change accoridngly if the size of embedding is not 50
    return vectors_dc

def get_sentence_embedding(text_data, vectors_dc):
    sentence_vectors=[]
    
    for sen in text_data:
        tokens=sen.split(' ')
        current_vector=np.array([vectors_dc[tokens[0]] if tokens[0] in vectors_dc else vectors_dc['<UNK>']])
        for word in tokens[1:]:
            if word in vectors_dc:
                current_vector=np.vstack([current_vector,vectors_dc[word]])
            else:
                current_vector=np.vstack([current_vector,vectors_dc['<UNK>']])
        min_max_mean=np.hstack([current_vector.min(axis=0),current_vector.max(axis=0),current_vector.mean(axis=0)])
        sentence_vectors.append(min_max_mean)

    return sentence_vectors


# Model Training

def heldout_score(clf, X_test, y_test,n_estimators =20):
    """compute deviance scores on ``X_test`` and ``y_test``. """
    score = np.zeros((n_estimators,), dtype=np.float64)
    for i, y_pred in enumerate(clf.staged_decision_function(X_test)):
        score[i] = clf.loss_(y_test, y_pred)
    return score

def cv_estimate(n_splits,X_train, y_train,n_estimators =20):
    best_score, best_model= None,None
    cv = KFold(n_splits=n_splits)
    cv_clf = GradientBoostingClassifier(n_estimators=n_estimators,min_samples_leaf=3, verbose=1, loss='deviance')
    val_scores = np.zeros((n_estimators,), dtype=np.float64)
    i=0
    for train, test in cv.split(X_train, y_train):
        cv_clf.fit(X_train[train], y_train[train])
        current_score= heldout_score(cv_clf, X_train[test], y_train[test],n_estimators)
        val_scores += current_score
        print ('Fold {} Score {}'.format(i+1, np.mean(current_score)))
        if i==0:
            best_score=np.mean(current_score)
            best_model=cv_clf
        else:
            if np.mean(current_score)<best_score:
                best_score=np.mean(current_score)
                best_model=cv_clf
        i+=1
    val_scores /= n_splits
    return val_scores, best_model

print ('Step1: Loading Training data')
train_texts=read_data(data_dir+'/training_text.csv')
train_labels=read_labels(data_dir+'/training_label.csv')

print ('Step 2 : Load word vectors')
vectors_dc=load_word_embedding(vectors_file)
len(vectors_dc)

print ('Step 3: Convert Word vectors to sentence vectors')
train_sentence_vectors=get_sentence_embedding(train_texts,vectors_dc)
print (len(train_sentence_vectors), len(train_labels), len(train_texts))

print ("Encoding data")
train_x, valid_x, train_y, valid_y=train_test_split(train_sentence_vectors, train_labels, test_size=0.2, random_state=random_seed)
train_x=np.array(train_x).astype('float32')
valid_x=np.array(valid_x).astype('float32')
train_y=np.array(train_y)
valid_y=np.array(valid_y)

print ('Step 4: Gradient Boosting Module using sklearn')
n_splits=3
cv_score,best_model = cv_estimate(n_splits,train_x, train_y, num_estimators)

run_logger.log("Num Iters vs. CV Score", cv_score)

print ('Step 5: Save the model')
model_identifier = 'evaluation_word2vec_gbm'
joblib.dump(best_model, models_dir+'//'+model_identifier)