from __future__ import absolute_import
from __future__ import division
import collections
import math
import os
import sys
import random
import numpy as np
from six.moves import urllib
from six.moves import xrange  
import tensorflow as tf
from timeit import default_timer as timer
import pandas as pd
import re
import io
from nltk.tokenize import TweetTokenizer
import num2words

training_filename = r'C:\Users\ds1\Documents\AzureML\data\training_text.csv'
model_identifier = 'Word2Vec_Basic'
embedding_folder = 'vectors'

if not os.path.exists(embedding_folder):
    os.makedirs(embedding_folder)

batch_size = 128      # Batch size
embedding_size = 50   # Dimension of the embedding vector.
skip_window = 2       # How many words to consider left and right.
num_skips = 2         # How many times to reuse an input to generate a label.

valid_size = 16     # Random set of words to evaluate similarity on.
valid_window = 100  # Only pick dev samples in the head of the distribution.
valid_examples = np.random.choice(valid_window, valid_size, replace=False)
num_sampled = 64    # Number of negative examples to sample.


# Data processing and batch preparation
# In the following code, we replace Emails, URLS, emoticons etc with special labels

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
    """Read the raw tweet data from a file. Replace Emails etc with special tokens """
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
        padded_data=' '.join(line for line in padded_lines)
        encoded_data=tf.compat.as_str(padded_data).split()    
    return encoded_data


def build_dataset(words, n_words):
    """Get raw tokens and build count dictionaries etc"""
    count = [['<UNK>', -1]]
    count.extend(collections.Counter(words).most_common(n_words))
    dictionary = {}
    reverse_dictionary={}
    for word, _ in count:
        temp = len(dictionary)
        dictionary[word]=temp
        reverse_dictionary[temp]=word
    data = list()
    unk_count = 0
    for word in words:
        if word in dictionary:
            index = dictionary[word]
        else:
            index = 0  # dictionary['UNK']
            unk_count += 1
        data.append(index)
    count[0][1] = unk_count
    return data, count, dictionary, reverse_dictionary

def generate_batch(data,batch_size, num_skips, skip_window ):
    """generate batches of data"""
    global data_index
    assert batch_size % num_skips == 0
    assert num_skips <= 2 * skip_window
    
    batch = np.ndarray(shape=(batch_size), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    span = 2 * skip_window + 1  # [ skip_window target skip_window ]
    buffer = collections.deque(maxlen=span)
    for _ in range(span):
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)
    for i in range(batch_size // num_skips):
        target = skip_window  # target label at the center of the buffer
        targets_to_avoid = [skip_window]
        for j in range(num_skips):
            while target in targets_to_avoid:
                target = random.randint(0, span - 1)
            targets_to_avoid.append(target)
            batch[i * num_skips + j] = buffer[skip_window]
            labels[i * num_skips + j, 0] = buffer[target]
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)
    data_index = (data_index + len(data) - span) % len(data)
    return batch, labels


# Model Training
# use device='/gpu:0' to train over gpu

def init_model(device='/cpu:0'):
    """ initialize model over the input device"""
   
    graph = tf.Graph()
    with graph.as_default():
        # Input data.
        train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
        train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
        valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

        with tf.device(device):
                # Look up embeddings for inputs.
                embeddings = tf.Variable(tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
                embed = tf.nn.embedding_lookup(embeddings, train_inputs)
                # Construct the variables for the NCE loss
                nce_weights = tf.Variable(tf.truncated_normal([vocabulary_size, embedding_size],
                                                              stddev=1.0 / math.sqrt(embedding_size)))
                nce_biases = tf.Variable(tf.zeros([vocabulary_size]))

        # Compute the average NCE loss for the batch.
        # tf.nce_loss automatically draws a new sample of the negative labels each
        # time we evaluate the loss.
        loss = tf.reduce_mean(
          tf.nn.nce_loss(weights=nce_weights,
                         biases=nce_biases,
                         labels=train_labels,
                         inputs=embed,
                         num_sampled=num_sampled,
                         num_classes=vocabulary_size))

        # Construct the SGD optimizer using a learning rate of 1.0.
        optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)

        # Compute the cosine similarity between minibatch examples and all embeddings.
        norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
        normalized_embeddings = embeddings / norm
        valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings, valid_dataset)
        similarity = tf.matmul(valid_embeddings, normalized_embeddings, transpose_b=True)
    
        # Add variable initializer.
        init = tf.global_variables_initializer()
        return graph, init, train_inputs, train_labels, valid_dataset, loss, optimizer,similarity,normalized_embeddings

def training(graph, init, train_inputs, train_labels, valid_dataset,loss, optimizer,similarity,normalized_embeddings):  
    """ train model over the tweet data"""
    num_steps = 100001

    with tf.Session(graph=graph) as session:
      # We must initialize all variables before we use them.
        init.run()
        print('Initialized')

        average_loss = 0
        for step in xrange(num_steps):
            batch_inputs, batch_labels = generate_batch(data,batch_size, num_skips, skip_window)
            feed_dict = {train_inputs: batch_inputs, train_labels: batch_labels}

            # We perform one update step by evaluating the optimizer op (including it
            # in the list of returned values for session.run()
            _, loss_val = session.run([optimizer, loss], feed_dict=feed_dict)
            average_loss += loss_val

            if step % 2000 == 0:
                if step > 0:
                    average_loss /= 2000
                  # The average loss is an estimate of the loss over the last 2000 batches.
                print('Average loss at step ', step, ': ', average_loss)
                average_loss = 0

            # Note that this is expensive (~20% slowdown if computed every 500 steps)
            if step % 10000 == 0:
                sim = similarity.eval()
                for i in xrange(valid_size):
                    valid_word = reverse_dictionary[valid_examples[i]]
                    top_k = 8  # number of nearest neighbors
                    nearest = (-sim[i, :]).argsort()[1:top_k + 1]
                    log_str = 'Nearest to %s:' % valid_word
                    for k in xrange(top_k):
                        close_word = reverse_dictionary[nearest[k]]
                        log_str = '%s %s,' % (log_str, close_word)
                #print(log_str)
        final_embeddings = normalized_embeddings.eval()
    return final_embeddings


# Export Embedding

def export_embeddings(final_embeddings):
    """ export embeddings to file"""
    embedding_weights=pd.DataFrame(final_embeddings).round(6).reset_index()
    word_indices_df=pd.DataFrame.from_dict(reverse_dictionary,orient='index').reset_index()
    word_indices_df.columns=['index','word']
    print (word_indices_df.shape,embedding_weights.shape)
    merged=pd.merge(word_indices_df,embedding_weights).drop('index',axis=1)
    print (merged.shape)
    merged.to_csv(embedding_folder+'//embeddings_{}.tsv'.format(model_identifier), sep='\t', index=False, header=False,
                 float_format='%.6f')
    return merged

print ('Step 1: Read and preprocess data')
vocabulary = read_data(training_filename)
vocabulary_size=len(set(vocabulary)) # Not excluding anything from the library # add 1 voc size
print ('Data size', len(vocabulary))
print ('Vocabulary Size', vocabulary_size +1) # including the count of Unknown i.e. UNK token
data, count, dictionary, reverse_dictionary = build_dataset(vocabulary, vocabulary_size)
del vocabulary  # Hint to reduce memory.
data_index = 0

print ('Step 2: Generate batches')
batch, labels = generate_batch(data,batch_size=8, num_skips=2, skip_window=1)

print ('Step 3 Actual Training')
start=timer()
graph, init, train_inputs, train_labels, valid_dataset,loss, optimizer,similarity,normalized_embeddings=init_model()
final_embedding=training(graph, init, train_inputs, train_labels, valid_dataset,loss, optimizer,similarity,normalized_embeddings)
end=timer()
print ('Total time taken to generate embedding' , (end-start))

print ('Step 4: Export Embeddings')
embedding_df=export_embeddings(final_embedding)