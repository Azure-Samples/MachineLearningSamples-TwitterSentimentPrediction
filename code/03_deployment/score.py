import numpy as np
import logging, sys, json
from keras.models import load_model
from DataPreparation import Data_Preparation
from sklearn.externals import joblib

#vectors_file='pickle_embeddings_Word2Vec_Basic.tsv'
#trainedModelFile="evaluation_word2vec_gbm"

vectors_file='pickle_embeddings_SSWE_Basic_Keras_w_CNTK.tsv'
trainedModelFile="evaluation_SSWE_logistic"

trainedModel = None
mem_after_init = None
labelLookup = None
topResult = 3

def init():
    """ Initialise SD model"""
    global trainedModel, labelLookup, mem_after_init, vector_size, reader,vectors_file
    vector_size = 50
    reader = Data_Preparation(vectors_file)
    # Load model and load the model from brainscript (3rd index)
    try:
        trainedModel = joblib.load(trainedModelFile)
    except:
        trainedModel=load_model(trainedModelFile)
        pass
    
def run(input_df):
    """ Classify the input using the loaded model"""
    global trainedModel
    import json
    # Generate Predictions	
    line=input_df.iloc[0]['input_text_string']
    #test_x=reader.get_sentence_embedding(['<BOS> '+line+' <EOS>'])
    test_x=reader.get_sentence_embedding([''+line+''])
    predictions = trainedModel.predict(test_x[0].flatten().reshape(1,150))
    y_pred = np.argmax(predictions, axis=1)
    y_pred_pos = predictions[:,1][0]
    print(y_pred_pos)	
    return (json.dumps(str(y_pred_pos)))