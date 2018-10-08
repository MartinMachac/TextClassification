# Transfer Learning in Text Classification

The aim of this project is to compare different approaches for text articles classification especially to try Transfer learning based on the model available as a part of TensorFlow.

## Dataset

##### 20 News groups dataset
The data is already split into training and testing part and is available as a part of scikit-learn. The training dataset contains 11314 records and test dataset 7532 records.

http://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html

## Pretrained text embedding model

##### TensorFlow's universal-sentence-encoder-large

>The Universal Sentence Encoder encodes text into high dimensional vectors that can be used for text classification, semantic similarity, clustering and other natural language tasks.
The model is trained and optimized for greater-than-word length text, such as sentences, phrases or short paragraphs. It is trained on a variety of data sources and a variety of tasks with the aim of dynamically accommodating a wide variety of natural language understanding tasks. The input is variable length English text and the output is a 512 dimensional vector. The universal-sentence-encoder-large model is trained with a Transformer encoder




## How to run it

### Text embedding

##### 20Newsgroups_embed_text.ipynb

Run this notebook to produce 512 dimensional vectors representing text data from the source dataset.
You can skip this step and use csv files provided for both training and testing data.

### Model training and validation

#### Features extration
    ##### TextClassification.ipynb

    Features extraction is done using CountVectorizer and TfidfTransformer from scikit-learn library.
    The final training dates is a combination of TFIDF extracted features and 512 dimensional vector got from text embedding

    ##### TextClassification_keras.ipynb

    Features extraction is done using Keras' Tokenizer
    The final training dates is a combination of TFIDF extracted features and 512 dimensional vector got from text embedding

#### Models Training

The training date were split into two parts 80 % training and 20 % for validation.
Two different feature sets were tested one based only of features extraction using Keras or scikit-learn and another where vectors with text embedding were added.

Support Vector Machines
Very fast training with reasonably good results with accuracy >84%.

Deep Neural Networks
Slower training (>20 times slower than SVM) slightly better accuracy with about 2% difference to SVM.



| Algo | Features |Accuracy
| ------ | ------ |
| SVM | scikit |83.6%
| SVM | keras |84.1%
| SVM | keras + embedding |83.8%
| DNN | scikit |84.1 %
| DNN | keras |84.9%
| DNN | keras + embedding |85.7 %
| DNN | scikit + embedding |86.4 %

Models hyper parameters optimization has not been done to some extent but there is very likely still some space from improvement. Especially with DNN it can be time consuming.

You can compare the results with other similar experiments

| Name | Details |Accuracy
| ------ | ------ |
| javedsha | SVM |82.3%
| Stanford | stanford_classifier |81.1%
| MS Research | SCDV |84.6%

References:

https://github.com/javedsha/text-classification
https://nlp.stanford.edu/wiki/Software/Classifier/20_Newsgroups
https://arxiv.org/pdf/1612.06778.pdf
