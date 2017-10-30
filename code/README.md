# Use word embedding to predict Twitter sentiment

## Introduction
In this tutorial we are going to demonstrate the usage of Word Embedding algorithms like **Word2Vec** algorithm and **Sentiment Specfic Word Embedding (SSWE) Algorithm** to predict Twitter sentiment in Azure Machine Learning Workbench.

For more information about Word2Vec and SSWE, you can refer to those papers: [Mikolov, Tomas, et al. Distributed representations of words and phrases and their compositionality. Advances in neural information processing systems. 2013.](https://arxiv.org/abs/1310.4546) and **Sentiment Specfic Word Embedding (SSWE) Algorithm** [Tang, Duyu, et al. "Learning Sentiment-Specific Word Embedding for Twitter Sentiment Classification." ACL (1). 2014.](http://www.aclweb.org/anthology/P14-1146) 

## Content
This tutorial consists of the following three main parts with each part consisting of one or more python scripts.

1. [Data Preparation](http://aka.ms/) 
2. [Modeling](https://aka.ms/) 
    * Feature Engineering
    * Model Creation
    * Model Evaluation 
3. [Deployment](http://aka.ms/) 

## Step-by-Step walkthrough

### Pre-requisite
Before diving into the project, some pre-requisites have to be met

- Set up Azure Subscription and Account
- Install Azure ML Work Bench
- Install some required packages

### Data Preparation
The first step in this tutorial is to download the sentiment140 dataset and divide it into train and test datasets. This part of the tutorial performs the downloading of the data and the splitting of data into train and test datasets. Execute 01_DownloadData.py in Azure ML Workbench Command Line to prepare the training and testing data. Remember to change the path of where the data set will be located. 

Sentiment140 dataset contains the actual content of the tweet (with emoticons removed) along with the polarity of each of the tweet (negative=0, neutral =2, positive=4) as well. Sentiment140 dataset has been labelled using the concept of distant supervision as explained in the paper **[Twitter Sentiment Classification Using Distant Supervision](http://cs.stanford.edu/people/alecmgo/papers/TwitterDistantSupervision09.pdf)**

Though the sentiment 140 dataset is internally divided into train and test subsets, the size of the test dataset is very small as compared to the train dataset. So, we are randomly split the training data into training and testing datasets.

![Data Preparation](../docs/media/01_DownloadData.PNG)

After this step is finished, several CSV files are generated in your specified data directory.

![Data Generated](../docs/media/02_DataSaved.PNG)

### Modeling
This part of the tutorial is further divided into three subparts: 
- **Feature Engineering** corresponds to the generation of features using different word embedding algorithms. 
- **Model Creation** deals with the training of different models like _logistic regression_ and _gradient boosting_ to predict sentiment of the input text. 
- **Model Evaluation** applies the trained model over the testing data.

#### Feature Engineering
We use Word2Vec and SSWE to generate word embeddings. 

##### Word2Vec
First we use the Word2Vec algorithm in the Skipgram mode as explained in the paper [Mikolov, Tomas, et al. Distributed representations of words and phrases and their compositionality. Advances in neural information processing systems. 2013.](https://arxiv.org/abs/1310.4546) to generate word embeddings.

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
<tr><td><img src="../docs/media/embedding_model2.PNG" alt="Skip-gram model"/></td></tr>
</table>

Modify the training file path in the scripts and run [02_A_Word2Vec.py](../02_modeling/02_A_Word2Vec.py) and [02_B_SSWE_Keras_w_CNTK.py](../02_modeling/02_B_SSWE_Keras_w_CNTK.py) in Azure ML Workbench CLI. 

First we train Word2Vec model to get the embedding, as the training process proceeds, the average loss is decreasing. After 100000 iterations, the loss is stable and training process is paused. Final embedding file has 304416 vocabularies, each of which has a embedding vector of size 50.

![Word2Vec](../docs/media/03_Word2Vec_training_1.PNG)
![Word2Vec](../docs/media/04_Word2Vec_training_2.PNG)

Then we train SSWE embedding model. 

![SSWE_training](../docs/media/06_SSWE_using_GPU.PNG)

After the training process is done, two embedding files in the format of TSV are generated in the vectors folder under 02_modeling.

![Embedding_Files](../docs/media/07_SSWE_Embedding_Basic_Keras_w_CNTK_TSV.PNG)

#### Model Creation
Once the word vectors have been generated using either of the SSWE or Word2vec algorithm, the next step is to train the classification models to predict actual sentiment polarity. However, before the actual training of the models the word level vectors have to be converted into sentence level vectors. The sentence level vectors are generated in two steps. In the first step, vectors of all the constituent words of a sentence are stacked up to get a matrix of size __maxsequencelength*embeddingdimension__. In the next step, min max and average operations are performed on each of the column of this matrix, hence resulting into a vector of size __3__ * __embeddingdimension__ for each of the sentence. 

This vector representation of sentences is given as input to the training classifiers. For this purpose of this tutorial, we demonstrate that how these sentence vectors can be used as input by simple linear models like Logistic Regression (using a single layer neural network in Keras) or the gradient boosting model based on sklearn. We have used 3-fold cross-validation in each of the notebook to select the best model. More details can be found in the individual notebooks in the [directory](./Code/02_Modeling/02_ModelCreation).

We apply the two types of features: Word2Vec and SSWE into two models: GBM model and Logistic regression model. Therefore we have four models to compare.

* Word2Vec in GBM model
![Word2Vec_GBM](../docs/media/08_ModelCreation_Word2Vec_GBM_training.PNG)

* Word2Vec in Logistic model
![Word2Vec_logit](../docs/media/08_B_ModelCreation_Word2Vec_Keras_training.PNG)
![Word2Vec_logit](../docs/media/08_C_ModelCreation_Word2Vec_Keras_training_done.PNG)

* SSWE in GBM model
![SSWE_GBM](../docs/media/10_B_ModelCreation_SSWE_GBM_training.PNG)
![SSWE_GBM](../docs/media/10_B_ModelCreation_SSWE_GBM_training_done.PNG)

* SSWE in Logistic model
![SSWE_logit](../docs/media/09_ModelCreation_SSWE_Keras_training.PNG)
![SSWE_logit](../docs/media/10_ModelCreation_SSWE_Keras_training_done.PNG)


#### Model Evaluation
We use the 4 trained in previous step in tetsting data to get evaluate the model's performance, GBM model with SSWE features is the best one in terms of AUC value.

1. Gradient Boosting over SSWE embedding
2. Logistic Regression over SSWE embedding
3. Gradient Boosting over Word2Vec embedding
4. Logistic Regression over Word2Vec embedding

![Compare_model](../docs/media/12_C_ModelEvaluation_Combined.PNG)


### Deployment
This part we will deploy pre-trained sentiment prediction model to a web service using Azure ML CLI. Several files are needed before deploying your model. Please move all those files under the project root directoty of Azure ML Work Bench.

* Pickled word embedding file
* Pre-trained model
* Scoring script
* Dependency yaml
* Model input in Json format

1. Pickle word embedding by running [pickle_embedding.py](../code/03_deployment/pickle_embedding.py), the resulting pickled word embedding files start with **pickle_**.

    ![pickled_file](../docs/media/13_Pickle_Two_Embedding_TSV.PNG)

2. Execute [schma_gen.py](../code/03_deployment/schema_gen.py) to create the schema required for web service, you will get a json file like this:

    ![schema_gen](../docs/media/15_schema_gen_SSWE_content.PNG)

3. Log in Azure account by running **az login** in Azure Machine Learning Work Bench Command Line.Follows the instructions on screen to login to your Azure account.

    ![open_cli](../docs/media/open_AML_CLI.PNG)

    ![az_login](../docs/media/17_az_login.PNG)

4. Set up Web service cluster using the following commands:
            
        az ml env setup -c -n <yourclustername> --location <e.g. eastus2>

    ![env_setup](../docs/media/18_az_ml_env_setup.PNG)

5. Set up Azure ML model management account(one time setup)

    az ml account modelmanagement create --location (e.g. eastus2) -n (your-new-acctname) -g (yourresourcegroupname) --sku-instances 1 --sku-name S1

    ![model_management_setup](../docs/media/19_az_ml_account_modelmanagement_setup.PNG)

6. Check cluster creation status using the this command, the creation may take several minutes to finish.

    az ml env show -g (yourresourcegroupname) -n (your-new-acctname)

    ![cluster_status](../docs/media/21_env_cluster_created.PNG)

7. Set deployment cluster

    az ml env set -n (yourclustername) -g (yourresourcegroupname) 

    ![set_env](../docs/media/22_az_ml_env_set_cluster.PNG)

8. You can check Kubernetes dashboard in the local host from your browser

    ![kubenetes_dashboard](../docs/media/25_kubernetes_dashboard.PNG)

9. Create realtime web service

    az ml service create realtime --model-file (model file name) -f (scoring script name) -n (your-new-acctname) -s (web service schema json file) -r (compute environment, python or PySpark, etc) -d (dependency files)

    ![create_realtime_webservice](../docs/media/23_create_realtime_webservice.PNG)

10. Check the status and usage of your realtime service

    az ml service show realtime -i <yourserviceid>

    ![realtime_service_usage](../docs/media/24_check_realtime_webservice.PNG) 

11.  Now you are ready to make prediction calls to web service 

    az ml service run realtime -i <yourserviceid> -d (web service input schema)

    ![make_prediction](../docs/media/26_call_realtime_service.PNG)


Congratulations! You have successfully deployed your model to a real time web service!