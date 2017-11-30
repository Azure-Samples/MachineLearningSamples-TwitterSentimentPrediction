# The purpose of this notebook is to train a **Logistic Regression** model using Keras to classify the tweets' sentiment as positive or negative.

import numpy as np
import pandas as pd
import os
import io

random_seed=1
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

# Path of the training file'
base_path = os.environ['HOMEPATH']
data_folder='data'
data_dir = os.path.join(base_path, data_folder)

# Path of the word vectors
embedding_folder = os.path.join(base_path, 'vectors')
vectors_file = os.path.join(embedding_folder, 'embeddings_Word2Vec_Basic.tsv')

model_identifier='evaluation_word2vec_logistic'

models_dir = os.path.join(base_path, 'model')

if not os.path.exists(models_dir):
    os.makedirs(models_dir)


# # Data Preprocessing

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
    """
    Read the raw tweet data from a file. Replace Emails etc with special tokens
    """
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
    """ read the tweet labels from the file
    """
    arr= np.genfromtxt(filename, delimiter='\n')
    arr[arr==4]=1 # Encode the positive category as 1
    return arr


# # Convert Word Vectors to Sentence Vectors

# The embeddings generated by both SSWE and Word2Vec algorithms are at word level but as we are using the sentences as the input, the word embeddings need to be converted to the sentence level embeddings. We are converting the word embeddings into sentence embeddings by using the approach in the original SSWE paper i.e. stacking the word vectors into a matrix and applying min, max and average operations on each of the columns of the word vectors matrix.

def load_word_embedding(vectors_file):
    """ Load the word vectors"""
    vectors= np.genfromtxt(vectors_file, delimiter='\t', comments='#--#',dtype=None,
                           names=['Word']+['EV{}'.format(i) for i in range(1,51)])#51 is embedding length + 1, change accoridngly if the size of embedding is not 50
    vectors_dc={}
    for x in vectors:
        vectors_dc[x['Word'].decode('utf-8','ignore')]=[float(x[each]) for each in ['EV{}'.format(i) for i in range(1,51)]]#51 is embedding length + 1, change accoridngly if the size of embedding is not 50
    return vectors_dc

def get_sentence_embedding(text_data, vectors_dc):
    """ This function converts the vectors of all the words in a sentence into sentence level vectors"""
    """ This function stacks up all the words vectors and then applies min, max and average operations over the stacked vectors"""
    """ If the size of the words vectors is n, then the size of the sentence vectors would be 3*n"""
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

# # Model Training


batch_size = 1028*6 # Batch Size should be changed according to the system specifications to have better utilization of GPU
nb_epoch = 30
    
def init_model():
    output_dim = no_classes = len(to_categorical(train_y)[0])
    input_dim=150
    model = Sequential() 
    model.add(Dense(output_dim, input_dim=input_dim, activation='softmax',activity_regularizer=regularizers.l1_l2(1))) 
    return model

def cv_estimate(n_splits,X_train, y_train):
    best_score, best_model= None,None
    cv = KFold(n_splits=n_splits)
    
    i=0
    for train, test in cv.split(X_train, y_train):
        model=init_model()
        mcp = ModelCheckpoint('./model_chkpoint_{}'.format(i), monitor="val_acc",
                      save_best_only=True, save_weights_only=False)
        model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy']) 

        model.fit(X_train[train], to_categorical(y_train[train]),epochs=nb_epoch,batch_size=batch_size, callbacks=[mcp],
                  validation_split=0.2)
        current_score= model.evaluate(X_train[test], to_categorical(y_train[test]))[0] # Getting the loss
        print ('\n Fold {} Current score {}'.format(i+1, current_score))
        
        
        if i==0:
            best_score=current_score
            best_model=model
        else:

            if current_score<best_score:
                best_score=current_score
                best_model=model
        i+=1

    return  best_model


# # Main

print ('Step 1: Loading Training data')
train_texts=read_data(data_dir+'/training_text.csv')
train_labels=read_labels(data_dir+'/training_label.csv')

print ("Step 2: Load Word Vectors")
vectors_dc=load_word_embedding(vectors_file)
len(vectors_dc)

print ("Step 3: Converting the word vectors to sentence vectors")
train_sentence_vectors=get_sentence_embedding(train_texts,vectors_dc)

print (" Encoding the data")
train_x=train_sentence_vectors
train_y=train_labels
train_x=np.array(train_x).astype('float32')
train_y=np.array(train_y)
print (len(train_sentence_vectors), len(train_labels), len(train_texts))

print ('Step 4: Logistic regression model using Keras')
best_model=cv_estimate(3,train_x, train_y)

print ("Step 5: Saving the model")
best_model.save(models_dir+'//'+model_identifier)