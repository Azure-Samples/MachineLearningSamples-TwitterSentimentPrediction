
# Using word embedding to predict Twitter Text Sentiment


## Background

Sentiment analysis is a widely research topic in the Natural Language Processing domain. It has the applications in consumer reviews mining, public opinion mining and advertisement on online forums. Many of the sentiment analysis approaches use handcrafted features but the popularity of unsupervised and semi supervised approached to generate word embeddings have made these embedding techniques an important way to generate features. In this tutorial we are going to demonstrate the usage of Word Embedding algorithms like **Word2Vec** algorithm [Mikolov, Tomas, et al. Distributed representations of words and phrases and their compositionality. Advances in neural information processing systems. 2013.](https://arxiv.org/abs/1310.4546) and **Sentiment Specfic Word Embedding (SSWE) Algorithm** [Tang, Duyu, et al. "Learning Sentiment-Specific Word Embedding for Twitter Sentiment Classification." ACL (1). 2014.](http://www.aclweb.org/anthology/P14-1146) for the purpose of sentiment polarity prediction using [Azure Machine Learning Workbench](https://docs.microsoft.com/en-us/azure/machine-learning/preview/overview-what-is-azure-ml).






## Content
This tutorial consists of the following three main parts with each part consisting of one or more Jupyter notebooks.

1. [Pre-requisite]
2. [Data Preparation](./Code/01_DataPreparation) 
3. [Modeling](./Code/02_Modeling/) 
    1. [Feature Engineering](./Code/02_Modeling/01_FeatureEngineering/)
    2. [Model Creation](./Code/02_Modeling/02_ModelCreation)
    3. [Model Evaluation](./Code/02_Modeling/03_ModelEvaluation) 
4. [Deployment](./Code/02_Modeling/03_Deployment) 

A brief description of each of these steps is as follows

### Pre-requisite
Before diving into the project, some pre-requisites have to be met

- Set up Azure Subscription and Account
- Install Azure ML Work Bench
- Install some required packages

### Data Preparation
The first step in this tutorial is to download the sentiment140 dataset and divide it into train and test datasets. This part of the tutorial performs the downloading of the data and the splitting of data into train and test datasets.

Sentiment140 dataset contains the actual content of the tweet (with emoticons removed) along with the polarity of each of the tweet (negative=0, neutral =2, positive=4) as well. Sentiment140 dataset has been labelled using the concept of distant supervision as explained in the paper **[Twitter Sentiment Classification Using Distant Supervision](http://cs.stanford.edu/people/alecmgo/papers/TwitterDistantSupervision09.pdf)**

Though the sentiment 140 dataset is internally divided into train and test subsets, the size of the test dataset is very small as compared to the train dataset. So, we are randomly split the training data into training and testing datasets.


### Modeling
This part of the tutorial is further divided into three subparts.
 
- **Feature Engineering** corresponds to the generation of features using different word embedding algorithms. 
- **Model Creation** deals with the training of different models like _logistic regression_ and _gradient boosting_ to predict sentiment of the input text. 
- **Model Evaluation** applies the trained model over the testing data
 
#### Feature Engineering
In this tutorial we have evaluated two different algorithms to generate word vectors (**Word2Vec** and **SSWE**) which are then used as input features for the classification algorithms.

##### Word2Vec
In this tutorial, we use the Word2Vec algorithm in the Skipgram mode as explained in the paper [Mikolov, Tomas, et al. Distributed representations of words and phrases and their compositionality. Advances in neural information processing systems. 2013.](https://arxiv.org/abs/1310.4546). 

Skip-gram is a shallow neural network taking the target word encoded as a one hot vector as input and using it to predict nearby words. If _V_ is the size of the vocabulary then the size of the output layer would be __C*V__ where C is the size of the context window. The skip-gram based architecture is shown in the following figure.
 
<table class="image" align="center">
<caption align="bottom">Skip-gram model</caption>
<tr><td><img src="https://s3-ap-south-1.amazonaws.com/av-blog-media/wp-content/uploads/2017/06/05000515/Capture2-276x300.png" alt="Skip-gram model"/></td></tr>
</table>

The details of the word2vec algorithm and skip-gram model are beyond the scope of this tutorial and the interested readers are requested to go through the following links for more details.

* [Vector Representation of Words](https://www.tensorflow.org/tutorials/word2vec)
* [How exactly does word2vec work?](http://www.1-4-5.net/~dmm/ml/how_does_word2vec_work.pdf)
* [Notes on Noise Contrastive Estimation and Negative Sampling](http://demo.clab.cs.cmu.edu/cdyer/nce_notes.pdf)


##### SSWE
**Sentiment Specfic Word Embedding (SSWE) Algorithm** proposed in [Tang, Duyu, et al. "Learning Sentiment-Specific Word Embedding for Twitter Sentiment Classification." ACL (1). 2014.](http://www.aclweb.org/anthology/P14-1146)  tries to overcome the weakness of Word2vec algorithm that the words with similar contexts and opposite polarity can have similar word vectors. This means that Word2vec may not perform very accurately for the tasks like sentiment analysis. SSWE algorithm tries to handle this weakness by incorporating both the sentence polarity and the word's context in to its loss function.

We are using a variant of SSWE in this tutorial. SSWE uses both the original ngram and corrupted ngram as input and it uses a ranking style hinge loss function for both the syntactic loss and the semantic loss. Ultimate loss function is the weighted combination of both the syntactic loss and semantic loss. For the purpose of simplicity, we are using only the semantic cross entropy as the loss function. As we are going to see later on, even with this simpler loss function the performance of the SSWE embedding is better than the Word2Vec embedding.

SSWE inspired neural network model that we use in this tutorial is shown in the following figure
<table class="image" align="center">
<caption align="bottom">Convolutional model to generate sentiment specific word embedding</caption>
<tr><td><img src="./Images/embedding_model2.png" alt="Skip-gram model"/></td></tr>
</table>


#### Model Creation

Once the word vectors have been generated using either of the SSWE or Word2vec algorithm, the next step is to train the classification models to predict actual sentiment polarity. However, before the actual training of the models the word level vectors have to be converted into sentence level vectors. The sentence level vectors are generated in two steps. In the first step, vectors of all the constituent words of a sentence are stacked up to get a matrix of size __maxsequencelength*embeddingdimension__. In the next step, min max and average operations are performed on each of the column of this matrix, hence resulting into a vector of size __3__ * __embeddingdimension__ for each of the sentence.
This vector representation of sentences is given as input to the training classifiers. For this purpose of this tutorial, we demonstrate that how these sentence vectors can be used as input by simple linear models like Logistic Regression (using a single layer neural network in Keras) or the gradient boosting model based on sklearn. We have used 3-fold cross-validation in each of the notebook to select the best model. More details can be found in the individual notebooks in the [directory](./Code/02_Modeling/02_ModelCreation).


#### Model Evaluation
In this step the models trained in the model creation step are applied on the test data that was created during the data preparation. We have evaluated four different models namely
1. Logistic Regression over SSWE embedding
2. Gradient Boosting over SSWE embedding
3. Logistic Regression over Word2Vec embedding
4. Gradient Boosting over Word2Vec embedding

AUC over test data for each of these experiments is as shown in the following figure.

<table class="image" align="center">
<caption align="bottom">Results</caption>
<tr><td><img src="./Images/results.png" alt="Results"/></td></tr>
</table>




### Deployment
This part of the tutorial demonstrates how to use Azure container services to deploy pre-trained sentiment prediction models.

These series of jupyter notebooks is self contained as all of the steps from the downloading of data to modeling and deployment are done by different blocks of code in the notebooks. The required packages to run each of notebook are specified at the start of each individual notebook. 


