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

The aim of this real-world scenario is to highlight how to use Azure Machine Learning Workbench to solve a complicated NLP task such as predict the sentiment of a given text. Here are the key points addressed:

1. How to train a Word2Vec embeddings model using Twitter text data
2. How to train a SSWE embeddings model using Twitter text data
2. How to build a deep Long Short-Term Memory (LSTM) recurrent neural network model for entity extraction on a GPU-enabled Azure Data Science Virtual Machine (GPU DSVM) on Azure.
2. Demonstrate that domain-specific word embeddings models can outperform generic word embeddings models in the entity recognition task. 
3. Demonstrate how to train and operationalize deep learning models using Azure Machine Learning Workbench.

The following capabilities within Azure Machine Learning Workbench:

   * Instantiation of [Team Data Science Process (TDSP) structure and templates](how-to-use-tdsp-in-azure-ml.md).
   * Automated management of your project dependencies including the download and the installation. 
   * Execution of code in Jupyter notebooks as well as Python scripts.
   * Run history tracking for Python files.
   * Execution of jobs on remote Spark compute context using HDInsight Spark 2.1 clusters.
   * Execution of jobs in remote GPU VMs on Azure.
   * Easy operationalization of deep learning models as web-services hosted on Azure Container Services.

The detailed documentation for this scenario including the step-by-step walk-through: https://review.docs.microsoft.com/en-us/azure/machine-learning/preview/scenario-tdsp-biomedical-recognition.

For code samples, click the View Project icon on the right and visit the project GitHub repository.

## Key components needed to run this example:

* An Azure [subscription](https://azure.microsoft.com/en-us/free/)
* Azure Machine Learning Workbench with a workspace created. See [installation guide](quick-start-installation.md). 
* To run this scenario with Spark cluster, provision [Azure HDInsight Spark cluster](https://docs.microsoft.com/en-us/azure/hdinsight/hdinsight-apache-spark-jupyter-spark-sql) (Spark 2.1 on Linux (HDI 3.6)) for scale-out computation. To process the full amount of MEDLINE abstracts discussed below, we recommend having a cluster with:
    * a head node of type [D13_V2](https://azure.microsoft.com/en-us/pricing/details/hdinsight/) 
    * at least four worker nodes of type [D12_V2](https://azure.microsoft.com/en-us/pricing/details/hdinsight/). 

    * To maximize performance of the cluster, we recommend to change the parameters spark.executor.instances, spark.executor.cores, and spark.executor.memory by following the instructions [here](https://docs.microsoft.com/en-us/azure/hdinsight/hdinsight-apache-spark-jupyter-spark-sql) and editing the definitions in "custom spark defaults" section. 

* You can run the entity extraction model training locally on a [Data Science Virtual Machine (DSVM)](https://docs.microsoft.com/en-us/azure/machine-learning/machine-learning-data-science-linux-dsvm-intro) or in a remote Docker container in a remote DSVM.

* To provision DSVM for Linux (Ubuntu), follow the instructions [here](https://docs.microsoft.com/en-us/azure/machine-learning/machine-learning-data-science-provision-vm). We recommend using [NC6 Standard (56 GB, K80 NVIDIA Tesla)](https://docs.microsoft.com/en-us/azure/machine-learning/machine-learning-data-science-linux-dsvm-intro).




# Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.microsoft.com.

When you submit a pull request, a CLA-bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., label, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.
