# Use word embedding to predict text sentiment


## Background

Sentiment analysis is a widely research topic in the Natural Language Processing domain. It has the applications in consumer reviews mining, public opinion mining and advertisement on online forums. Many of the sentiment analysis approaches use handcrafted features but the popularity of unsupervised and semi supervised approached to generate word embeddings have made these embedding techniques an important way to generate features. In this tutorial/ series of notebooks we are going to demonstrate the usage of Word Embedding algorithms like **Word2Vec** algorithm [Mikolov, Tomas, et al. Distributed representations of words and phrases and their compositionality. Advances in neural information processing systems. 2013.](https://arxiv.org/abs/1310.4546) and **Sentiment Specfic Word Embedding (SSWE) Algorithm** [Tang, Duyu, et al. "Learning Sentiment-Specific Word Embedding for Twitter Sentiment Classification." ACL (1). 2014.](http://www.aclweb.org/anthology/P14-1146) for the purpose of sentiment polarity prediction.

## Content
This tutorial consists of the following three main parts with each part consisting of one or more Jupyter notebooks.

1. [Data Preparation](http://aka.ms/) 
1. [Modeling](https://aka.ms/) 
    1. Feature Engineering
    1. Model Creation
    1. Model Evaluation 
1. [Deployment](http://aka.ms/) 

A brief description of each of these steps is as follows

### Data Preparation
The first step in this tutorial is to download the sentiment140 dataset and divide it into train and test datasets. This part of the tutorial performs the downloading of the data and the splitting of data into train and test datasets

### Modeling
This part of the tutorial is further divided into three subparts. **Feature Engineering** corresponds to the generation of features using different word embedding algorithms. ** Model Creation ** deals with the training of different models like _logistic regression_ and _gradient boosting_ to predict sentiment of the input text. **Model Evaluation** applies the trained model over the testing data

### Deployment
This part of the tutorial demonstrates how to use Azure container services to deploy pre-trained sentiment prediction models.
