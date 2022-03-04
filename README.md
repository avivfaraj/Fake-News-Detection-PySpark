# Fake News Detection using PySpark

### by: Aviv Farag

---

## Table of Contents
 * [Abstract](#abstract)
 * [Dependencies](#dependencies)
 * [Functions](#functions)
 * [Custom Transformer](#custom-transformer)
 * [Setup and running the code](#setup-and-running-the-code)
 * [Acknowledgements](#acknowledgements)
 

---

## Abstract: 
Fake news is articles that contains misleading information aiming to change other’s opinion,  thus gaining power (political, business, etc.). In this study, I propose a machine learning model based on Naive Bayesand implemented in PySpark for classifying document into two groups of news: reliable and fake. Data cleaning,stop words removing, and counting terms frequency were all implemented to generate the training and test datasets.  Results of the ML model were compared to the baseline using confusion matrix, and revealed a great improvement in accuracy and F1 score.

---

## Dependencies:
1. NLTK

	```
    import nltk
    from nltk.stem import PorterStemmer
    from nltk.corpus import stopwords
	```

1. PySpark

	```
    from pyspark.sql import SparkSession
    from pyspark.sql.types import StructType,StructField, StringType, IntegerType
    from pyspark.ml.feature import IDF, Tokenizer, VectorAssembler
    from pyspark.ml.feature import StopWordsRemover, CountVectorizer
    from pyspark.ml import Pipeline, PipelineModel
    from pyspark.sql.functions import when, col, regexp_replace, concat, lit, length
    from pyspark.sql.types import FloatType, DoubleType
    from pyspark.ml.classification import NaiveBayesModel, NaiveBayes
    from pyspark.mllib.evaluation import BinaryClassificationMetrics
	```
  
1. Others
	```
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt

	```
---
## Functions

1. **evaluate(df, labelCol = "label", predCol = "prediction")** <br>
Compute precision, accuracy, F1 score, and recall. Print them as well as the confusion matrix, and return some of them as a tuple: (confusion_matrix, precision, recall)

---
## Custom Transformer:
**class Stemmer(Transformer, HasInputCol, HasOutputCol, DefaultParamsReadable, DefaultParamsWritable):** <br>
Convert every word in the list to its stem using the NLTK PorterStem instance, thus reducing the dimension of the features column. For example, the words "Playing", "Plays", and "Played" are all converted to "Play".

---
## Setup and running the code:
Running on Google Collaboratoy:
Upload *Final Project.ipynb* file to google drive, launch it using Google Collaboratoy, and follow the instructions

Any other platform:
Clone the repo using the following command in terminal:<br>
	`git clone https://github.com/avivfaraj/DSCI631-project.git`
	
Upload *Final Project.ipynb* and the dataset to the platform of your choice. Before running the code, make sure to change the path to dataset.

---

## Acknowledgements

Dataset was found at [Kaggle](https://www.kaggle.com/c/fake-news/data). <br>

