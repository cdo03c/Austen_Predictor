#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 09:44:55 2017

@author: obrien

Version: 1.0
Dependencies: Python 3.6.X, Keras 2.0.4, Numpy 1.13.3, Ipython 5.3.0
"""

### Modified from Essentials of Deep Learning : Introduction to Long Short Term Memory
### https://www.analyticsvidhya.com/blog/2017/12/fundamentals-of-deep-learning-introduction-to-lstm/?utm_source=feedburner&utm_medium=email&utm_campaign=Feed%3A+AnalyticsVidhya+%28Analytics+Vidhya%29

# Importing dependencies numpy and keras
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.models import load_model
import requests
import re
import os
import pandas as pd
import time

### FUNCTION DEFINITIONS ###
def get_text(urls):
    '''This function takes in a a list of urls strings and returns a
    single string with all the characters minus the header and escape
    characters.
    '''
    
    books = [requests.get(u).text.lower() for u in urls]

    #Remove of the Gutenberg website headers
    books = remove_header(books)

    ##Combine the the books into one continuous character string
    text = ''.join(books)

    #Use REGEX to remove escape characters from the text.
    return(re.sub('[\r\t\n\ufeff]', ' ', text))

def remove_header(books, stop_text = ' ***'):
    return([books[b][books[b].find(stop_text)+len(stop_text):] for b in range(len(books))])

def build_model(shape):
    
    # defining the LSTM model
    model = Sequential()
    model.add(LSTM(300, input_shape=(shape[1], shape[2]), return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(300))
    model.add(Dropout(0.2))
    model.add(Dense(shape[2], activation='relu'))

# =============================================================================
#     model.add(LSTM(
#         input_dim=1,
#         output_dim=shape[1],
#         return_sequences=True))
#     model.add(Dropout(0.2))
# 
#     model.add(LSTM(
#         layers[2],
#         return_sequences=False))
#     model.add(Dropout(0.2))
# 
#     model.add(Dense(output_dim=layers[3]))
# 
#     model.add(LSTM(
#         input_dim=layers[0],
#         output_dim=layers[1],
#         return_sequences=True))
#     model.add(Dropout(0.2))
# 
#     model.add(LSTM(
#         layers[2],
#         return_sequences=False))
#     model.add(Dropout(0.2))
# 
#     model.add(Dense(output_dim=layers[3]))
# =============================================================================

    start = time.time()
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    print("> Compilation Time : ", time.time() - start)
    return model

### MAINT SCRIPT ###
if(__name__ == "__main__"):
    
    global_start_time = time.time()

    #Sets parameters
    epochs  = 1
    seq_len = 50
    #Load path to the model
    #model_path = r'austen_model.h5'
    model_path = ''
    

    urls = ['https://www.gutenberg.org/files/141/141-0.txt',
            'https://www.gutenberg.org/files/121/121-0.txt',
            'https://www.gutenberg.org/cache/epub/105/pg105.txt',
            'https://www.gutenberg.org/cache/epub/161/pg161.txt',
            'https://www.gutenberg.org/files/158/158-0.txt',
            'https://www.gutenberg.org/files/1342/1342-0.txt']

    text = get_text(urls)

    print('Corpus loaded and cleaned')

    #Builds a list of all words and unique words
    words = re.compile('\w+').findall(text)
    unique_words = sorted(list(set(words)))

    #Builds word to integer mappings
    word_to_int = dict((w, i) for i, w in enumerate(unique_words))
    int_to_word = dict((i, w) for i, w in enumerate(unique_words))
    
    #Builds word sequences of the number of seq_len with the next word
    #as the label
    step = 5
    sequences = []
    labels = []
    for i in range(0, len(words) - seq_len, step):
        sequences.append(words[i: i + seq_len])
        labels.append(words[i + seq_len])
    print(f'num training examples: {len(sequences)}')
    
    
    X = np.zeros((len(sequences), seq_len, len(unique_words)), dtype=np.bool)
    y = np.zeros((len(sequences), len(unique_words)), dtype=np.bool)
    for i, sequence in enumerate(sequences):
        for t, word in enumerate(sequence):
            X[i, t, word_to_int[word]] = 1
        y[i, word_to_int[labels[i]]] = 1

    print('Training vectors generated')



    #Test if the model exists at the path saved above and loads the model if it
    #already exists.
    if os.path.exists(model_path):
        print('Loading Model')
        model = load_model(model_path)
    else:
        print('Training Model')
        model = build_model(X.shape)

        # fitting the model
        model.fit(X[0:10001], y[0:10001], epochs=1, batch_size=30,
                  validation_split = .1)

# =============================================================================
# WARNING - on the following system it took 89465 seconds (24.8 hours)
# to run.
#
#   Model Name:	MacBook Pro
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
        model.save(r'austen_model_step5.h5')



    test_num = 500
    rand_strings = [np.random.randint(0, len(X)-1) for i in range(test_num)]
    test_strings = [np.reshape(X[r], (1, 50, 1)) / float(len(unique_words)) for r in rand_strings]
    pred = [int_to_word[np.argmax(model.predict(s, verbose=0))] for s in test_strings]
    actual = [int_to_word[y[r]] for r in rand_strings]
    df = pd.DataFrame(actual, columns = ['actual'])
    df['prediction'] = pred
    print(df)
    print('Built test sequences')