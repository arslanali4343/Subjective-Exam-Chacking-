# 👋 Introducing `Subjective Exam-Chacking`  
## In this file I used 11 different Models.
## 1) Problem Statement
We have observed that many questions on web based question-answering/discussion platforms go unanswered for a long time. The main reason behind that is either the question is asked in the wrong category or the similar kind of question has been asked before So people tend not to answer it. That’s why the CrowdSource team at Google Research, a group dedicated to advancing NLP and other types of ML science via crowdsourcing, has collected data on a number of these quality scoring aspects. We use that dataset to build predictive algorithms for different subjective aspects of question-answering. The question-answer pairs were gathered from nearly 70 different websites, in a "common-sense" fashion. Our raters received minimal guidance and training and relied largely on their subjective interpretation of the prompts. As such, each prompt was crafted in the most intuitive fashion so that raters could simply use their common sense to complete the task. Demonstrating these subjective labels can be predicted reliably and can shine a new light on this research area.
## 2) PROPOSED SYSTEM

i) Types of Exam Supported:

    - Objective
    - Subjective
    - Practical 
    
ii) If webpage is refresh then the timer will not be refreshed

iii) Support for Negative Marking.

iv) Support for randomize questions.

v) Support for Calculator for Mathematical type of Exam

vi) Support for 20 types of Compilers/Interpreter for  programming practical type of Exam.

vii) For Objective type of Exam:

     - Single page per question
     - Bookmark question 
     - Question Grid with previous & next button
     - At the time of exam submission all questions statistics will be showed to user for confirmation. 

## 3) MODEL LINK YOLO V3:
https://pjreddie.com/media/files/yolov3.weights
## 4) DataSet Google QUEST Q&A Labeling
Improving automated understanding of complex question answer content

### 4(i) About this Competition
The data for this competition includes questions and answers from various StackExchange properties. Your task is to predict target values of 30 labels for each question-answer pair.

The list of 30 target labels are the same as the column names in the sample_submission.csv file. Target labels with the prefix question_ relate to the question_title and/or question_body features in the data. Target labels with the prefix answer_ relate to the answer feature.

Each row contains a single question and a single answer to that question, along with additional features. The training data contains rows with some duplicated questions (but with different answers). The test data does not contain any duplicated questions.

This is not a binary prediction challenge. Target labels are aggregated from multiple raters, and can have continuous values in the range [0,1]. Therefore, predictions must also be in that range.


### 4(ii) File descriptions
train.csv - the training data (target labels are the last 30 columns)
test.csv - the test set (you must predict 30 labels for each test set row)
## 5) Libraries
![image](https://user-images.githubusercontent.com/53410060/193464453-f62575e5-cf58-4807-ba80-99f21f9e3308.png)



      
## 6) Data Pre-Processing
### 6(i) Removing the URL's from the text
URLs (or Uniform Resource Locators) in a text are references to a location on the web, but do not provide any additional information. We thus, remove these too using the library named re, which provides regular expression matching operations.
### 6(ii) Removing the Tags from the text
The web generates tons of text data and this text might have HTML tags in it. These HTML tags do not add any value to text data and only enable proper browser rendering. Hence we will remove the HTML tags from the text using re library
### 6(iii) Lowercasing the text
The generated text contains both uppecase characters as well as lower case characters. Systems are usually case sensitive so it would consider "the" and "The" as different word, which would not only increase the number of words we have process but also cause same word to have multiple meaning. Hence we will lower case the entire text
### 6(iv) Expand contracted words in the text
In our everyday verbal and written communication, a lot of us tend to contract common words like “you are” becomes “you’re”. Converting contractions into their natural form will bring more insights.
### 6(v) Remove words with numbers
The words which contain number tend to be spam, and add more noise to the data. Hence we'll remove them
### 6(vi) Remove special characters
Special characters like – (hyphen) or / (slash) don’t add any value, so we generally remove those. Characters are removed depending on the use case. If we are performing a task where the currency doesn’t play a role (for example in sentiment analysis), we remove the any currency sign.
## 7) NLP STEPS
### 7(i) Stop Word Removal
Apart from URLs, HTML tags and special characters, there are words that are not required for tasks such as sentiment analysis or text classification. Words like I, me, you, he and others increase the size of text data but don’t improve results dramatically and thus it is a good idea to remove those.

Instead of going with standard NLTK stopword set we decided to make our own set, as other set also includes negative words like 'not' which could be useful for the task
### 7(ii) Lemmatization
Now that we have removed all the “noise” from the text, it is time to normalize the data set. A word in a text may exist in multiple forms like stop and stopped (past participle or price and prices (plural). Text normalization converts variations of the word into root form of the same word.
## 8) Building Different Models
### 8(ii) Building a Bare Minimal Neural Network
Here, we will be building a stand-alonw neural network model just for classifying the labels to the respective questions. For this we will be using RNNs/LSTMs/GRU for our usecase. A classic LSTM based network is one of the most fundamental building blocks of all the robust architectures that we see today.For the first part we will be focussing on standard RNNs. Some resources for RNNs:

### 8(iii) Recurrent Neural Networks
Recurrent neural networks (RNN) are a class of neural networks that is powerful for modeling sequence data such as time series or natural language.Schematically, a RNN layer uses a for loop to iterate over the timesteps of a sequence, while maintaining an internal state that encodes information about the timesteps it has seen so far. Forward pass of Classical RNNs have the following formula :

### 8(iv) Classical RNN image
A classic RNN consists of the following image:
![image](https://user-images.githubusercontent.com/53410060/193461123-612fad42-9bce-409f-bc91-5282012d3068.png)
### 8(v) Simple RNN
Model Architecture
The model architecture for the Bidirectional Simple RNN can be seen as below:

### 8(vi) Embedding Matrix using GloVe Word Embeddings


### 8(vii) Simple RNN with Glove200D pretrained embeddings
Total params: 1,033,341

Trainable params: 1,033,341

Non-trainable params: 0

### 8(viii) LSTM- Long Short Term Memory
LSTMs are gated recurrent networks having 4 gates with (tanh/sigmoid) activation units. These architectures are the the building blocks of all the transformer architectures that we see, and the 4 gates combine input from different time stamps to produce the output. In a LSTM, there are typically 3 input and output signals: The h (hidden cell output from the previous timestep), c (the signal from previous cell), and the x(input vectors). Outputs involve the updated ht+1(hidden cell output of current block) value, ct+1, (updated c signal from the present cell) and the output(o).
![image](https://user-images.githubusercontent.com/53410060/193461576-51a78a2d-bb65-4bf9-8cff-fd9fbf325307.png)
### 8(ix) Gated Recurrent Units
GRUs A slightly more dramatic variation on the LSTM is the Gated Recurrent Unit, or GRU, introduced by Cho, et al. (2014). It combines the forget and input gates into a single “update gate.” It also merges the cell state and hidden state, and makes some other changes. The resulting model is simpler than standard LSTM models, and has been growing increasingly popular.

![image](https://user-images.githubusercontent.com/53410060/193461654-4d55c976-0045-45f2-ab81-ba62cfadca9a.png)

### 8(x) Universal Sentence Encoder
We'll download the Universal Sentence Encoder model from tensorflow hub and use the same to obtain the embeddings for titles of all the question answer pairs.
![image](https://user-images.githubusercontent.com/53410060/193461709-0b120f51-e82f-4ebc-9db2-79078f867ef8.png)
### 8(xi) Semantic Similarity Based Retrival
We'll fin6 d the cosine similarity of query with every every question title and return the question title with maximum similarity

## IF YOU ARE LOOKING ONLY FOR PROCTORING REFER THIS LIBRARY:
## Package Link: https://pypi.org/project/proctoring/
## Github Link: https://github.com/narender-rk10/proctoring

## LICENSE:
<a rel="license" href="http://creativecommons.org/licenses/by-nc-nd/4.0/"><img alt="Creative Commons Licence" style="border-width:0" src="https://i.creativecommons.org/l/by-nc-nd/4.0/88x31.png" /></a><br />This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-nc-nd/4.0/">Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License</a>.
