# -*- coding: utf-8 -*-
"""
Created on Mon Mar 12 11:01:05 2018

@author: KIMJIHAE
"""
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Convolution2D, Dense, Dropout, Flatten, MaxPooling2D
import theano
import keras.backend as K

K.set_image_dim_ordering('tf')
theano.config.optimizer = "None"


# input 128*128
# filter 64 128 256
# mask size 5*5
# max pooling 2*2
# fully connected 500 300(softmax)
def CNN(weights_path=None):
    cnn = Sequential()
    cnn.add(Convolution2D(64, (5, 5), padding="same", activation="relu", input_shape=(128, 128, 1)))
    cnn.add(MaxPooling2D(pool_size=(2, 2)))

    cnn.add(Convolution2D(128, (5, 5), padding="same", activation="relu"))
    cnn.add(MaxPooling2D(pool_size=(2, 2)))

    cnn.add(Convolution2D(256, (5, 5), padding="same", activation="relu"))
    cnn.add(MaxPooling2D(pool_size=(2, 2)))

    cnn.add(Flatten())
    cnn.add(Dense(1024, activation="relu", name='fc4'))
    cnn.add(Dropout(0.5))
    cnn.add(Dense(500, activation="relu", name='fc5'))
    cnn.add(Dropout(0.5))
    cnn.add(Dense(6, activation="softmax", name='fc6'))

    if weights_path:
        # cnn.load_weights(weights_path)
        cnn = load_model(weights_path)

    return cnn