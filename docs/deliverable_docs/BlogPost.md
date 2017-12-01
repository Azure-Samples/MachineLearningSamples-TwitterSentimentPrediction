# Use word embeddings to predict Twitter sentiment following Team Data Science Process

The primary purpose of this sample is to show how to instantiate and execute a complex machine learning project using the [Team Data Science Process (TDSP)](https://docs.microsoft.com/en-us/azure/machine-learning/team-data-science-process/overview) structure and templates in [Azure Machine Learning Workbench](https://docs.microsoft.com/en-us/azure/machine-learning/preview/). 

In this walkthrough, we demonstrate the usage of Word Embedding algorithms like Word2Vec algorithm and Sentiment Specfic Word Embedding (SSWE) Algorithm to predict Twitter sentiment polarity in Azure Machine Learning Workbench. The trained model is deployed to a web service using Azure Container Service(ACS). We are following Team Data Science Process to execute this project.

This project is executed using [TDSP templates]((https://github.com/Azure/Microsoft-TDSP/blob/master/Docs/lifecycle-detail.md)) which consist of the following parts: 

- Data acquisition and understanding
- Modeling
    - Feature Engineering
    - Model Creation
    - Model Evaluation
- Deployment

Some highlights from this sample:
- This sample is running on the latest [Azure Machine Learning Workbench](https://docs.microsoft.com/en-us/azure/machine-learning/preview/), which is currently in public preview.
- Model training is performed in [Azure Data Science Virtual Machine with GPU]((https://docs.microsoft.com/en-us/azure/machine-learning/machine-learning-data-science-linux-dsvm-intro)).
- Word embeddings using Word2Vec and SSWE are generated for modeling.
- Deep learning frameworks and packages such as TensorFlow, CNTK and Keras are applied in this project.
- Four models using different word embedding methods and modeling techniques are trained and compared.
- The trained model is deployed to a web service using [Azure Container Service](https://azure.microsoft.com/en-us/services/container-service/).

The Word2Vec algorithm is based on the the paper [Mikolov, Tomas, et al. Distributed representations of words and phrases and their compositionality. Advances in neural information processing systems. 2013.](https://arxiv.org/abs/1310.4546). Skip-gram is a shallow neural network taking the target word encoded as a one hot vector as input and using it to predict nearby words. The skip-gram based architecture is shown in the following figure.
 
<table class="image" align="center">
<caption align="bottom">Skip-gram model</caption>
<tr><td><img src="https://s3-ap-south-1.amazonaws.com/av-blog-media/wp-content/uploads/2017/06/05000515/Capture2-276x300.png" alt="Skip-gram model"/></td></tr>
</table>

Sentiment Specfic Word Embedding (SSWE) Algorithm proposed in [Tang, Duyu, et al. "Learning Sentiment-Specific Word Embedding for Twitter Sentiment Classification." ACL (1). 2014.](http://www.aclweb.org/anthology/P14-1146) tries to overcome the weakness of Word2vec algorithm that the words with similar contexts and opposite polarity can have similar word vectors. This means that Word2vec may not perform very accurately for the tasks like sentiment analysis. SSWE algorithm tries to handle this weakness by incorporating both the sentence polarity and the word's context in to its loss function.

We are using a variant of SSWE in this sample. SSWE uses both the original ngram and corrupted ngram as input and it uses a ranking style hinge loss function for both the syntactic loss and the semantic loss. Ultimate loss function is the weighted combination of both the syntactic loss and semantic loss. For the purpose of simplicity, we are using only the semantic cross entropy as the loss function. As we are going to see later on, even with this simpler loss function the performance of the SSWE embedding is better than the Word2Vec embedding. SSWE inspired neural network model that we use in this sample is shown in the following figure
<table class="image" align="center">
<caption align="bottom">Convolutional model to generate sentiment specific word embedding</caption>
<tr><td><img src="../deliverable_docs/images/embedding_model2.PNG" alt="Skip-gram model"/></td></tr>
</table>

<table class="image" align="center">
<caption align="bottom">Model Comparison</caption>
<tr><td><img src="../deliverable_docs/images/model_comparison.PNG" width="450" height="350" alt="Skip-gram model"/></td></tr>
</td></tr>
</table>

For details, please refer to this [article](https://docs.microsoft.com/azure/machine-learning/preview/scenario-tdsp-twitter-sentiment). 
We also provided all the scripts and a detailed [walkthrough](https://github.com/Azure/MachineLearningSamples-TwitterSentimentPrediction/blob/master/docs/deliverable_docs/Step_By_Step_Tutorial.md) in this [GitHub repository](https://github.com/Azure/MachineLearningSamples-TwitterSentimentPrediction). 

We would love to hear your feedback on this sample – you can send us your feedback and comments via the GitHub [issues page](https://github.com/Azure/MachineLearningSamples-TwitterSentimentPrediction/issues).





