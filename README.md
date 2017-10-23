# Using word embedding to predict Twitter Text Sentiment


## Link to the Microsoft DOCS site

The detailed documentation for this example includes the step-by-step walk-through:
[https://docs.microsoft.com/azure/machine-learning/preview/scenario-tdsp-twitter-sentiment](https://docs.microsoft.com/azure/machine-learning/preview/scenario-tdsp-twitter-sentiment)


## Link to the Gallery GitHub repository

The public GitHub repository for this example contains all the code samples:
[https://github.com/Azure/MachineLearningSamples-TwitterSentimentPrediction](https://github.com/Azure/MachineLearningSamples-TwitterSentimentPrediction)


## Summary

Sentiment analysis is a widely research topic in the Natural Language Processing domain. It has the applications in consumer reviews mining, public opinion mining and advertisement on online forums. Many of the sentiment analysis approaches use handcrafted features but the popularity of unsupervised and semi supervised approached to generate word embeddings have made these embedding techniques an important way to generate features. In this tutorial we are going to demonstrate the usage of Word Embedding algorithms like **Word2Vec** and **SSWE** to predict sentiment polarity. This end-to-end process is implemented in [Azure Machine Learning Workbench](https://docs.microsoft.com/en-us/azure/machine-learning/preview/overview-what-is-azure-ml).


## Description

The aim of this tutorial is to highlight how to use Azure Machine Learning Workbench to predict the sentiment of Twitter text data. Here are the key points addressed:

1. How to train a Word2Vec embeddings model
2. How to train a SSWE embeddings model
2. How to use Word2Vec and SSWE embeddings in GBM model and Logistic Model in Keras using CNTK/TensorFlow backend on a GPU-enabled Azure Data Science Virtual Machine (GPU DSVM).
2. Demonstrate that GBM model using SSWE embeddings achieves the best model in terms of AUC
3. Demonstrate how to train and operationalize a machine learning model using Azure Machine Learning Workbench.

The following capabilities within Azure Machine Learning Workbench are covered in this tutorial:

   * Instantiation of [Team Data Science Process (TDSP) structure and templates](how-to-use-tdsp-in-azure-ml.md).
   * Automated management of your project dependencies including the download and the installation. 
   * Execution of Python scripts.
   * Run history tracking for Python files.
   * Execution of jobs in Azure GPU VMs.
   * Easy operationalization of learning models as web-services hosted on Azure Container Services.

The detailed documentation for this scenario including the step-by-step walk-through: https://review.docs.microsoft.com/en-us/azure/machine-learning/preview/scenario-tdsp-twitter-sentiment.

For code samples, click the View Project icon on the right and visit the project GitHub repository.

## Key components needed to run this example:

* An Azure [subscription](https://azure.microsoft.com/en-us/free/)
* Azure Machine Learning Workbench with a workspace created. See [installation guide](quick-start-installation.md). 
* You can run through the tutorial locally on a [Data Science Virtual Machine (DSVM)](https://docs.microsoft.com/en-us/azure/machine-learning/machine-learning-data-science-linux-dsvm-intro).
* To provision DSVM for Windows 2016, follow the instructions [here](https://docs.microsoft.com/en-us/azure/machine-learning/machine-learning-data-science-provision-vm). We recommend using [NC6 Standard (56 GB, K80 NVIDIA Tesla)](https://docs.microsoft.com/en-us/azure/machine-learning/machine-learning-data-science-linux-dsvm-intro).


# Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us the rights to use your contribution. For details, visit https://cla.microsoft.com.

When you submit a pull request, a CLA-bot will automatically determine whether you need to provide a CLA and decorate the PR appropriately (e.g., label, comment). Simply follow the instructions provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.
