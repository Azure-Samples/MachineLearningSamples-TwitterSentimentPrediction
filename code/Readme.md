# /code

This directory contains all the source code for the project. There are three subdirectories, adhering to the stages of the TDSP lifecycle.

The code sections are organized below in sequential order in which they are to be run. Before running code, do setup.


## /code/01\_data\_acquisition\_and\_understanding
This folder contains code for data preparation and exploratory analyses. It also contains any necessary settings files needed to run the data exploration code. 

Data preparation is performed using the 01_DataPreparation.py. In this file Twitter data is downloaded to local machine, tweets texts and labels are extracted.

Further details on the code used for data preparation and exploratory analysis is provided in [code/01\_data\_acquisition\_and\_understanding](https://github.com/Azure/MachineLearningSamples-TwitterSentimentPrediction/tree/master/code/01_data_acquisition_and_understanding).  


## code/02_modeling
This folder contains code related to modeling, including feature engineering, model creation, and model evaluation. 

In feature engineering stage, two types word embeddings are created: Word2Vec and SSWE.

In model creation stage, Logistic regression and Gradient Boosted Tree are created using the two types of word embeddings. There are four models created in this stage.

In model evaluation stage, the performance of above four models are compared using testing data, Gradient Boosted Tree model with SSWE embedding has the highest AUC value among the four models.

Detail about the code used in modeling is provided in [code/02_modeling](https://github.com/Azure/MachineLearningSamples-TwitterSentimentPrediction/tree/master/code/02_modeling).

## code/03_deployment
This folder contains code related to deployment of the Gradient Boosted Tree model with SSWE embedding in Azure Container Services. Detail about the code used in deployment is provided in [code/03_deployment](https://github.com/Azure/MachineLearningSamples-TwitterSentimentPrediction/tree/master/code/03_deployment).

## Execution
### Code run in local compute context
In this example, we execute code in **local compute environment** only. Refer to Azure Machine Learning documents for execution details and further options.

### Running .py files
Executing a Python script in a local Python runtime is easy:

    az ml experiment submit -c local your_python_file.py