import os
import pandas as pd
import numpy as np
from zipfile import ZipFile
import urllib.request
from tempfile import mktemp

base_path=r'C:\Users\ds1\Documents\AzureML'
base_folder='data'

# URL to download the sentiment140 dataset
data_url='http://cs.stanford.edu/people/alecmgo/trainingandtestdata.zip'

# Functions to download and process data

def change_base_dir(base_dir_path):
    """ Change the working directopry of the code"""
    
    if not os.path.exists(base_dir_path):
        print ('creating directory', base_dir_path)
        os.makedirs(base_dir_path)
    print ('Changing base directory to ', base_dir_path)
    os.chdir(base_dir_path)

def download_data(download_url, filename='downloaded_data.zip'):
    """ Download and extract data """
    
    downloaded_filename = os.path.join('.', filename)
    print ('Step 1: Downloading data')
    urllib.request.urlretrieve(download_url,downloaded_filename)
    print ('Step 2: Extracting data')
    zipfile=ZipFile(downloaded_filename)
    zipfile.extractall('./')
    zipfile.close()

def extract_tweets_and_labels(filename ):
    """ Extract tweets and labels from the downloaded data"""
    
    print ('Step 3: Reading the data as a dataframe')
    df=pd.read_csv(filename, header=None, encoding='iso-8859-1')    
    df.columns=['Label','TweetId','Date','Query','User','Text']
    print ('Read {} lines'.format(df.shape[0]))
    print ('Discarding neutral tweets')
    df=df[df.Label!=2]
    print ('No of lines in the data after filtering neutral tweets: {}'.format(df.shape[0]))
    print ('Step 4: Shuffling the data')
    train_length=int(df.shape[0]*0.8)    
    df=df.sample(frac=1) # reshuffling the data
      
    df['Text']=df['Text'].astype(str).apply(lambda x:x.strip())#.encode('ascii','ignore')#str.decode('utf8','ignore')#.str.encode('ascii','ignore')
    print (df.head())
    print ('Step 5: Dividing into test and train datasets')
    df_train = df.iloc[:train_length, :]
    df_test = df.iloc[train_length:, :]    
    
    print ('Step 6: Exporting the train and test datasets')    
    print ('Exporting training data of rows {}'.format(df_train.shape[0]))
    export_prefix='training'
    df_train[['Label']].to_csv(export_prefix+'_label.csv', header=False, index=False)
    df_train[['Text']].to_csv(export_prefix+'_text.csv', header=False, index=False)
    print ('Target distribution in the training data is as follows')
    print ('\n',df_train['Label'].value_counts()) 
    
    print ('Exporting training data of rows {}'.format(df_test.shape[0]))
    export_prefix='testing'
    df_test[['Label']].to_csv(export_prefix+'_label.csv', header=False, index=False)
    df_test[['Text']].to_csv(export_prefix+'_text.csv', header=False, index=False)
    print ('Target distribution in the testing data is as follows')
    print ('\n',df_test['Label'].value_counts())
    

# Download and processing the data

base_dir_path=base_path+'\\'+base_folder
change_base_dir(base_dir_path)
download_data(data_url)
extract_tweets_and_labels('training.1600000.processed.noemoticon.csv')
