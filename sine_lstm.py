#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 16 06:30:43 2018

@author: cdo03c
"""
### IMPORTING DEPENDENCIES ###
import numpy as np
from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout, LSTM
import time
import os
from numpy import newaxis
import matplotlib.pyplot as plt


### FUNCTION DEFINITIONS ###
def load_data(filename, seq_len, normalise_window = False):
    '''This function takes in a file path string, sequence lenth integer,
    and normalise window boolean, and reads in the data and returns a
    list of numpy arrays for training inputs (X_train), training outputs
    (y_train), testing inputs (X_test), and testing outputs (y_test).
    '''
    f = open(filename, 'rb').read()
    data = f.decode().split('\n')

    sequence_length = seq_len + 1
    result = [data[index: index + sequence_length] for index in range(len(data) - sequence_length)]
        
    if normalise_window:
        result = normalise_windows(result)

    result = np.array(result)

    row = round(0.9 * result.shape[0])
    train = result[:int(row), :]
    np.random.shuffle(train)
    x_train = train[:, :-1]
    y_train = train[:, -1]
    x_test = result[int(row):, :-1]
    y_test = result[int(row):, -1]

    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))  

    return [x_train, y_train, x_test, y_test]

def normalise_windows(window_data):
    normalised_data = []
    for window in window_data:
        normalised_window = [((float(p) / float(window[0])) - 1) for p in window]
        normalised_data.append(normalised_window)
    return normalised_data

def build_model(layers):
    model = Sequential()

    model.add(LSTM(
        input_dim=layers[0],
        output_dim=layers[1],
        return_sequences=True))
    model.add(Dropout(0.2))

    model.add(LSTM(
        layers[2],
        return_sequences=False))
    model.add(Dropout(0.2))

    model.add(Dense(
        output_dim=layers[3]))
    model.add(Activation("linear"))

    start = time.time()
    model.compile(loss="mse", optimizer="rmsprop")
    print("> Compilation Time : ", time.time() - start)
    return model

def predict_point_by_point(model, data):
    #Predict each timestep given the last sequence of true data, in effect only predicting 1 step ahead each time
    predicted = model.predict(data)
    predicted = np.reshape(predicted, (predicted.size,))
    return predicted

def predict_sequence_full(model, data, window_size):
    #Shift the window by 1 new prediction each time, re-run predictions on new window
    curr_frame = data[0]
    predicted = []
    for i in range(len(data)):
        predicted.append(model.predict(curr_frame[newaxis,:,:])[0,0])
        curr_frame = curr_frame[1:]
        curr_frame = np.insert(curr_frame, [window_size-1], predicted[-1], axis=0)
    return predicted

def plot_results(true_data, predicted_data = np.array([])):
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(true_data, label='True Data')
    if predicted_data.size > 0:
        plt.plot(predicted_data, label='Prediction')
    plt.legend()
    plt.show()

### MAIN SCRIPT ###
    
if(__name__ == "__main__"):
    path = os.path.dirname(os.path.abspath(__file__))
    
    global_start_time = time.time()
    epochs  = 1
    seq_len = 50

    print('> Loading data... ')

    #The load_data function works if the sinwave.csv is in the same directory
    #as the sine_lstm.py script.
    X_train, y_train, X_test, y_test = load_data(path + '/sinwave.csv',
                                                 seq_len)
    
    print('> Data Loaded. Compiling...')
    
    model = build_model([1, 50, 100, 1])
    
    model.fit(
	    X_train,
	    y_train,
	    batch_size=512,
	    nb_epoch=epochs,
	    validation_split=0.05)
    
    #predictions = lstm.predict_sequences_multiple(model, X_test, seq_len, 50)
    #predicted = predict_sequence_full(model, X_test, seq_len)
    predicted = predict_point_by_point(model, X_test) 
    
    print('Training duration (s) : {}'.format(time.time() - global_start_time))
    
    plot_results(y_test)
    plot_results(y_test, predicted)
    