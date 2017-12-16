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
import requests
import re

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

# defining the LSTM model
model = Sequential()
model.add(LSTM(300, input_shape=(X_modified.shape[1], X_modified.shape[2]), return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(300))
model.add(Dropout(0.2))
model.add(Dense(Y_modified.shape[1], activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam')

# fitting the model
model.fit(X_modified, Y_modified, epochs=1, batch_size=30)

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

# picking a random seed
start_index = numpy.random.randint(0, len(X)-1)
new_string = X[start_index]

# generating words
for i in range(50):
    x = numpy.reshape(new_string, (1, len(new_string), 1))
    x = x / float(len(unique_words))
    
#Build sentences for prediction
sentences = text.split('.')

#predicting
pred_index = numpy.argmax(model.predict(x, verbose=0))
word_out = int_to_word[pred_index]
seq_in = [int_to_word[value] for value in new_string]
print(word_out)

new_string.append(pred_index)
new_string = new_string[1:len(new_string)]