from keras import backend as K
import os

def set_keras_backend(backend):
    if K.backend() != backend:
        os.environ['KERAS_BACKEND'] = backend
        try:
            from importlib import reload
            reload(K)  # Python 2.7
        except NameError:
            try:
                from importlib import reload  # Python 3.4+
                reload(K)
            except ImportError:
                from imp import reload  # Python 3.0 - 3.3
                reload(K)
        assert K.backend() == backend

set_keras_backend("cntk")
K.set_image_dim_ordering('tf')

import pandas as pd
import numpy as np
from timeit import default_timer as timer
from keras.callbacks import ModelCheckpoint
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Input, Dense, Flatten, Embedding
from keras.layers.pooling import GlobalMaxPooling1D,MaxPooling1D
from keras.layers.convolutional import Convolution1D
from keras.layers.core import Lambda
from keras import optimizers
from keras.models import Model
from keras.regularizers import l1
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from sklearn import svm, metrics
from sklearn.model_selection import train_test_split
from IPython.display import SVG
import pydot
from keras.utils.vis_utils import model_to_dot
import re
import io
from nltk.tokenize import TweetTokenizer
from nltk.tokenize import RegexpTokenizer
import num2words

random_seed=1
np.random.seed(random_seed)

data_dir = r'C:\Users\ds1\Documents\AzureML\data'
embedding_folder = 'vectors'
model_identifier = 'SSWE_Basic_Keras_w_CNTK'

if not os.path.exists(embedding_folder):
    os.makedirs(embedding_folder)

max_sequence_length = 15 # each sentence of the input should be padded to have at least this many tokens
embedding_dim 		= 50 # Embedding layer size
no_filters			= 15 # No of filters for the convolution layer
filter_size			= 5  # Filter size for the convolution layer
trainable 			= True # flag specifying whether the embedding layer weights should be changed during the training or not
batch_size 			= 128 # batch size can be increased to have better gpu utilization
no_epochs 			= 5 # No of training epochs

# Data preprocessing

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

emoticonsDict = {}
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

# Loading Training and Validation Data
texts 				= []
labels 				= []
nb_train_samples	= 0
nb_valid_samples 	= 0

print ('Loading Training Labels')
train_labels=read_labels(data_dir+'\\training_label.csv')

print ('Loading Training data')
train_texts=read_data(data_dir+'//training_text.csv')

print (len(train_labels), len(train_texts))
print ("Using Keras tokenizer to tokenize and build word index")
tokenizer = Tokenizer(lower=False, filters='\n\t?"!') 
train_texts=[each for each in train_texts]
tokenizer.fit_on_texts(train_texts)
sorted_voc = [wc[0] for wc in sorted(tokenizer.word_counts.items(),reverse=True, key= lambda x:x[1]) ]
tokenizer.word_index = dict(list(zip(sorted_voc, list(range(2, len(sorted_voc) + 2)))))
tokenizer.word_index['<PAD>']=0
tokenizer.word_index['<UNK>']=1
word_index = tokenizer.word_index
reverse_dictionary={v:k for (k,v) in tokenizer.word_index.items()}
vocab_size=len(tokenizer.word_index.keys())

print ('Size of the vocab is', vocab_size)


# Shuffling /Padding the data

print ('Padding sentences and shuffling the data')
sequences = tokenizer.texts_to_sequences(train_texts)

#Pad the sentences to have consistent length
data = pad_sequences(sequences, maxlen=max_sequence_length, padding='post')
labels = to_categorical(np.asarray(train_labels))
indices = np.arange(len(labels))
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]

train_x, valid_x, train_y, valid_y=train_test_split(data, labels, test_size=0.2, random_state=random_seed)
train_x=np.array(train_x).astype('float32')
valid_x=np.array(valid_x).astype('float32')
train_y=np.array(train_y)
valid_y=np.array(valid_y)
embedding_matrix = np.zeros((len(word_index) , embedding_dim))
training_word_index=tokenizer.word_index.copy()

# Model Instantiation
print ('Initializing the model')
mcp = ModelCheckpoint('./model_chkpoint', monitor="val_acc", save_best_only=True, save_weights_only=False)

#Creating network
model = Sequential()
model.add(Embedding(len(word_index)+2,
                            embedding_dim,
                            input_length=max_sequence_length,
                            trainable=trainable, name='embedding'))
model.add(Convolution1D(no_filters, filter_size, activation='relu'))
model.add(MaxPooling1D(max_sequence_length - filter_size))
model.add(Flatten())
model.add(Dense(no_filters, activation='tanh'))
model.add(Dense(len(labels[0]), activation='softmax'))

optim=optimizers.Adam(lr=0.1, )
model.compile(loss='categorical_crossentropy',
              optimizer=optim,
              metrics=['acc'])
model.summary()

# Training
start=timer()
hist=model.fit(train_x, train_y,nb_epoch=no_epochs, batch_size=batch_size,validation_data=(valid_x, valid_y),callbacks=[mcp])
end=timer()

# Exporting the Embedding Matrix and Vocabulary
def export_embeddings(model_orig):
    """ export embeddings to file"""
    embedding_weights=pd.DataFrame(model_orig.layers[0].get_weights()[0]).reset_index()
    word_indices_df=pd.DataFrame.from_dict(training_word_index,orient='index').reset_index()
    word_indices_df.columns=['word','index']
    print (word_indices_df.shape,embedding_weights.shape)
    merged=pd.merge(word_indices_df,embedding_weights)
    print (merged.shape)
    merged=merged[[each for each in merged.columns if each!='index']]    
    merged.to_csv(embedding_folder+'//embeddings_{}.tsv'.format(model_identifier), sep='\t', 
              index=False, header=False,float_format='%.6f',encoding='utf-8')
    return embedding_weights, word_indices_df, merged

embedding_weights, word_indices_df, merged_df=export_embeddings(model)