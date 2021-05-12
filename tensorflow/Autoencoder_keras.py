from keras.models import Model
from keras import optimizers
from keras.models import Sequential
from keras.layers import Layer, Convolution2D, Dense, Dropout, Flatten, MaxPooling2D, ZeroPadding2D, Reshape,UpSampling2D,BatchNormalization,Activation
import theano
import keras.backend as K
from os import environ
from importlib import reload

#from keras.models import load_model
import os.path
img_channels=1
img_rows=128
img_cols=128
def set_keras_backend(backend):
    if K.backend()!=backend:
        environ['KERAS_BACKEND']=backend
        reload(K)
        assert  K.backend()==backend
set_keras_backend('theano')
K.set_image_dim_ordering('th')
#x = Input(shape=original_img_size)
def Autoencoder(weights_path=None):
    encoding_layers = [
        ZeroPadding2D((1, 1)),
        Convolution2D(64, (3, 3), activation='relu'),
        ZeroPadding2D((1, 1)),
        Convolution2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2), strides=(2, 2)),

        ZeroPadding2D((1, 1)),
        Convolution2D(128, (3, 3), activation='relu'),
        ZeroPadding2D((1, 1)),
        Convolution2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2), strides=(2, 2)),

        ZeroPadding2D((1, 1)),
        Convolution2D(256, (3, 3), activation='relu'),
        ZeroPadding2D((1, 1)),
        Convolution2D(256, (3, 3), activation='relu'),
        ZeroPadding2D((1, 1)),
        Convolution2D(256, (3, 3), activation='relu'),
        ZeroPadding2D((1, 1)),
        Convolution2D(256, (3, 3), activation='relu'),
        MaxPooling2D((2, 2), strides=(2, 2)),

        Convolution2D(256, (3, 3), activation='relu'),
        MaxPooling2D((2, 2), strides=(2, 2)),
    ]

    decoding_layers = [
        UpSampling2D(size=(2, 2)),
        Convolution2D(256, (3, 3), padding='same'),
        BatchNormalization(),
        Activation('relu'),

        UpSampling2D(size=(2, 2)),
        Convolution2D(256, (3, 3), padding='same'),
        BatchNormalization(),
        Activation('relu'),

        Convolution2D(256, (3, 3), padding='same'),
        BatchNormalization(),
        Activation('relu'),

        Convolution2D(128, (3, 3), padding='same'),
        BatchNormalization(),
        Activation('relu'),

        UpSampling2D(size=(2, 2)),
        Convolution2D(128, (3, 3), padding='same'),
        BatchNormalization(),
        Activation('relu'),

        Convolution2D(64, (3, 3), padding='same'),
        BatchNormalization(),
        Activation('relu'),

        UpSampling2D(size=(2, 2)),
        Convolution2D(1, (3, 3), padding='same'),
        BatchNormalization(),
    ]
    model = Sequential()

    model.add(Layer(input_shape=(1,128,128)))

    model.encoding_layers = encoding_layers
    for l in model.encoding_layers:
        model.add(l)

    model.add(Flatten())
    model.add(Dense(256 * 8 * 8, activation='relu'))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(256 * 8 * 8, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Reshape((256, 8, 8), input_shape=(-1, 128 * 8 * 8,)))
    model.decoding_layers = decoding_layers
    for l in model.decoding_layers:
        model.add(l)

    model.add(Activation('sigmoid'))

    #segnet_basic.add(Permute((2, 1)))
    #segnet_basic.add(Activation('softmax'))
    if weights_path:
        model.load_weights(weights_path)

    return model