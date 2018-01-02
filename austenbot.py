#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 09:44:55 2017

@author: obrien
"""

### Modified from Essentials of Deep Learning : Introduction to Long Short Term Memory
### https://www.analyticsvidhya.com/blog/2017/12/fundamentals-of-deep-learning-introduction-to-lstm/?utm_source=feedburner&utm_medium=email&utm_campaign=Feed%3A+AnalyticsVidhya+%28Analytics+Vidhya%29

# Importing dependencies numpy and keras
import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.utils import np_utils 
from keras.models import load_model
import requests
import re
import os
import pandas as pd

#Download the text from the six Jane Austen novels and store in a list
#called books
urls = ['https://www.gutenberg.org/files/141/141-0.txt',
        'https://www.gutenberg.org/files/121/121-0.txt',
        'https://www.gutenberg.org/cache/epub/105/pg105.txt',
        'https://www.gutenberg.org/cache/epub/161/pg161.txt',
        'https://www.gutenberg.org/files/158/158-0.txt',
        'https://www.gutenberg.org/files/1342/1342-0.txt']
books = [requests.get(u).text.lower() for u in urls]

##Combine the the books into one continuous character string
text = ''
for b in books:
    text = text + b

#Use REGEX to remove escape characters from the text.
text = re.sub('[\r\t\n\ufeff]', ' ', text)

print('Corpus loaded and cleaned')

# mapping words with integers
words = re.compile('\w+').findall(text)
unique_words = sorted(list(set(words)))

word_to_int = {}
int_to_word = {}

for i, c in enumerate (unique_words):
    word_to_int.update({c: i})
    int_to_word.update({i: c})

# preparing input and output dataset
X = []
Y = []

for i in range(0, len(words) - 50, 1):
    sequence = words[i:i + 50]
    label =words[i + 50]
    X.append([word_to_int[char] for char in sequence])
    Y.append(word_to_int[label])
    
# reshaping, normalizing and one hot encoding
X_modified = numpy.reshape(X, (len(X), 50, 1))
X_modified = X_modified / float(len(unique_words))
Y_modified = np_utils.to_categorical(Y)

print('Training vectors generated')

#Load path to the model
model_path = r'austen_model.h5'

#Test if the model exists at the path saved above and loads the model if it
#already exists.
if os.path.exists(model_path):
    print('Loading Model')
    model = load_model(model_path)
else:
    print('Training Model')
    # defining the LSTM model
    model = Sequential()
    model.add(LSTM(300, input_shape=(X_modified.shape[1], X_modified.shape[2]), return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(300))
    model.add(Dropout(0.2))
    model.add(Dense(Y_modified.shape[1], activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam')

    # fitting the model
    model.fit(X_modified[0:10000], Y_modified[0:10000], epochs=1, batch_size=30)



# =============================================================================
# WARNING - on the following system it took 89465 seconds (24.8 hours)
# to run.
#
# Model Name:	MacBook Pro
#   Model Identifier:	MacBookPro11,1
#   Processor Name:	Intel Core i5
#   Processor Speed:	2.4 GHz
#   Number of Processors:	1
#   Total Number of Cores:	2
#   L2 Cache (per Core):	256 KB
#   L3 Cache:	3 MB
#   Memory:	8 GB
# =============================================================================

    #Save out model after fitting
    model.save(r'austen_model10000.h5')

# =============================================================================
# Below is code for generating random word sequences to feed into model for 
# prediction.
# This can be interesting, but this fails at the task of understanding whether
# or not the model is good at predicting what Jane Austen would have written
# so at first we take the corpus of her novels on which the model was trained
# and generate sentences to see if the model can predict the last word in senetences
# of at least 5 words.  Then Jane Austen's letters are loaded to understand
# how good the model is at predicting her words on previously unseen data.
#
# # picking a random seed
# start_index = numpy.random.randint(0, len(X)-1)
# new_string = X[start_index]
# 
# # generating random words
# for i in range(50):
#     x = numpy.reshape(new_string, (1, len(new_string), 1))
#     x = x / float(len(unique_words))
# #predicting
# pred_index = numpy.argmax(model.predict(x, verbose=0))
# word_out = int_to_word[pred_index]
# seq_in = [int_to_word[value] for value in new_string]
# print(seq_in, word_out)

# new_string.append(pred_index)
# new_string = new_string[1:len(new_string)]
# =============================================================================
    
##Build sentences for prediction by creating a list of sentence strings
##missing the last word from the texts and a list of the last words to
##use as ground truth for prediction.
#sentences = text.split('.')
#wordSentences = [re.compile('\w+').findall(s) for s in sentences 
#                 if len(re.compile('\w+').findall(s)) > 5]
#intSentences = [[word_to_int[word] for word in words] for words in wordSentences]
#testSentences = [ints[:-1] for ints in intSentences]
#testLabels = [ints[-1:] for ints in intSentences]
#
#x1 = numpy.reshape(testSentences[0], (1,len(testSentences[0]),1))
#x1 = x1 / float(len(unique_words))
#pred_index = numpy.argmax(model.predict(x, verbose=0))

test_num = 500
rand_strings = [numpy.random.randint(0, len(X)-1) for i in range(test_num)]
test_strings = [numpy.reshape(X[r], (1, 50, 1)) / float(len(unique_words)) for r in rand_strings]
pred = [int_to_word[numpy.argmax(model.predict(s, verbose=0))] for s in test_strings]
actual = [int_to_word[Y[r]] for r in rand_strings]
df = pd.DataFrame(actual, columns = ['actual'])
df['prediction'] = pred
print(df)
print('Built test sequences')