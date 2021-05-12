# -*- coding: utf-8 -*-
"""
Created on Tue Mar  6 16:49:43 2018

@author: KIMJIHAE
"""
from keras.models import Sequential
from keras.layers import Convolution2D, Dense, Dropout, Flatten, MaxPooling2D
from keras.utils import np_utils
import numpy as np
import theano
from train_app_6 import set_keras_backend
import keras.backend as K
set_keras_backend('tensorflow')
K.set_image_dim_ordering('tf')
def shallow_CNN(weights_path=None):
    #DATA LOAD
    '''
    # process the data to fit in a keras CNN properly
    # input data needs to be (N, C, X, Y) - shaped where
    # N - number of samples
    # C - number of channels per sample
    # (X, Y) - sample size    
    # output labels should be one-hot vectors - ie,
    # 0 -> [0, 0, 1]
    # 1 -> [0, 1, 0]
    # 2 -> [1, 0, 0]
    
    # define a CNN
    # see http://keras.io for API reference'''
    cnn = Sequential()
    cnn.add(Convolution2D(64, (3, 1),
        padding="same",
        activation="relu",
        input_shape=(36, 1, 1)))
    cnn.add(Convolution2D(64, (3, 1), padding="same", activation="relu"))
    cnn.add(MaxPooling2D(pool_size=(2,1)))
    
    cnn.add(Convolution2D(128, (3, 1), padding="same", activation="relu"))
    cnn.add(Convolution2D(128, (3, 1), padding="same", activation="relu"))
    cnn.add(Convolution2D(128, (3, 1), padding="same", activation="relu"))
    cnn.add(MaxPooling2D(pool_size=(2,1)))
        
    cnn.add(Convolution2D(256, (3, 1), padding="same", activation="relu"))
    cnn.add(Convolution2D(256, (3, 1), padding="same", activation="relu"))
    cnn.add(Convolution2D(256, (3, 1), padding="same", activation="relu"))
    cnn.add(MaxPooling2D(pool_size=(2, 1)))
        
    cnn.add(Flatten())
    cnn.add(Dense(500, activation="relu", name='fc4'))
    cnn.add(Dropout(0.5))
    cnn.add(Dense(2, activation="softmax", name='fc5'))
    
    if weights_path:
        cnn.load_weights(weights_path)

    return cnn