# -*- coding: utf-8 -*-
"""
Created on Sat Feb 24 17:04:58 2018

@author: KIMJIHAE
"""
import numpy as np
import os
from keras.utils import np_utils
from geo_cnn_keras import shallow_CNN
from keras.models import load_model
from itertools import combinations
import get_dataset as getdb
import convert_ckplus_files as ck
import convert_jaffe_files as jaffe
import convert_ferg_files as ferg
import geometric as geo
from keras.callbacks import ModelCheckpoint
import keras.backend as K
from sklearn.utils import shuffle
from train_cnn import get_n_dataset, loading_n_data
from util import count_dataset
dirname_ck = 'ckplus/extended-cohn-kanade-images/cohn-kanade-images/'
import keras.backend as K
from Autoencoder_keras import Autoencoder
import Autoencoder_main as auto_main
from os import environ
from importlib import reload
def set_keras_backend(backend):
    if K.backend() != backend:
        environ['KERAS_BACKEND'] = backend
        reload(K)
        assert K.backend() == backend
set_keras_backend('tensorflow')
K.set_image_dim_ordering('tf')
# loading data
print('Auto Model Loading...')
set_keras_backend('theano')
K.set_image_dim_ordering('th')
automodel = Autoencoder('Autoencoder/model/weights_best_169_0.03.hdf5')
print('Auto Model Loading Complete')
set_keras_backend('tensorflow')
K.set_image_dim_ordering('tf')
def loading_data(str,cycle): # cycle 1-10
    Y=[]
    filepath=[]
    ### 0:FERG, 1:JAFFE, 2:CK+, 3:AffectNet
    if int(str)==0:
        filepath=np.load('app_cnn_6/train_dataset/{0}/0_path_train_dataset.npy'.format((cycle)))
        Y=np.load('app_cnn_6/train_dataset/{0}/0_Y_train_dataset.npy'.format((cycle)))
    elif int(str)==1:
        filepath = np.load('app_cnn_6/train_dataset/{0}/1_path_train_dataset.npy'.format((cycle)))
        Y = np.load('app_cnn_6/train_dataset/{0}/1_Y_train_dataset.npy'.format((cycle)))

    elif int(str)==2:
        filepath = np.load('app_cnn_6/train_dataset/{0}/2_path_train_dataset.npy'.format((cycle)))
        Y = np.load('app_cnn_6/train_dataset/{0}/2_Y_train_dataset.npy'.format((cycle)))
    elif int(str) == 3:
        filepath = np.load('app_cnn_6/train_dataset/{0}/3_path_train_dataset.npy'.format((cycle)))
        Y = np.load('app_cnn_6/train_dataset/{0}/3_Y_train_dataset.npy'.format((cycle)))
        print('file path lenght:',filepath.shape)
    return filepath, Y

# loading data
def loading_data_test(str,cycle): # cycle 1-10
    Y=[]
    filepath=[]
    ### 0:FERG, 1:JAFFE, 2:CK+, 3:AffectNet
    if int(str) == 0:
        filepath=np.load('app_cnn_6/test_dataset/{0}/0_path_test_dataset.npy'.format((cycle)))
        Y=np.load('app_cnn_6/test_dataset/{0}/0_Y_test_dataset.npy'.format((cycle)))
    elif int(str) == 1:
        filepath = np.load('app_cnn_6/test_dataset/{0}/1_path_test_dataset.npy'.format((cycle)))
        Y = np.load('app_cnn_6/test_dataset/{0}/1_Y_test_dataset.npy'.format((cycle)))
    elif int(str) == 2:
        filepath = np.load('app_cnn_6/test_dataset/{0}/2_path_test_dataset.npy'.format((cycle)))
        Y = np.load('app_cnn_6/test_dataset/{0}/2_Y_test_dataset.npy'.format((cycle)))
    elif int(str) == 3:
        filepath = np.load('app_cnn_6/test_dataset/{0}/3_path_test_dataset.npy'.format((cycle)))
        Y = np.load('app_cnn_6/test_dataset/{0}/3_Y_test_dataset.npy'.format((cycle)))

    return filepath, Y

# loading geometric data
def loading_geo_data(filepath, Y, str): # cycle 1-10
    X_geo=[]
    Y_geo=[]

    if int(str) == 0:
        ferg_all_path = np.load('app_cnn_6/train_dataset/all_path_0_dataset.npy')
        Y1 = np.load('app_cnn_6/train_dataset/all_Y_0_dataset.npy')
        for i in range(0, len(filepath[0])):
            tmp = getdb.get_ferg_geo(filepath[0][i], ferg_all_path, Y1)
            if tmp is not -1:
                for j in range(0, len(tmp)):
                    X_geo.append(tmp[j])
                    Y_geo.append(Y[0][i])
    elif int(str) == 1:
        jaffe_all_path = np.load('app_cnn_6/train_dataset/all_path_1_dataset.npy')
        Y2 = np.load('app_cnn_6/train_dataset/all_Y_1_dataset.npy')
        for i in range(0, len(filepath[0])):
            print(filepath[0][i])
            tmp = getdb.get_jaffe_geo(filepath[0][i], jaffe_all_path, Y2)
            if tmp is not -1:
                for j in range(0, len(tmp)):
                    X_geo.append(tmp[j])
                    Y_geo.append(Y[0][i])
    elif int(str) == 2:
        ckplus_all_path = np.load('app_cnn_6/train_dataset/all_path_2_dataset.npy')
        Y3 = np.load('app_cnn_6/train_dataset/all_Y_2_dataset.npy')
        for i in range(0, len(filepath[0])):
            tmp = getdb.get_ck_plus_geo(filepath[0][i], ckplus_all_path, Y3)
            if tmp is not -1:
                for j in range(0, len(tmp)):
                    X_geo.append(tmp[j])
                    Y_geo.append(Y[0][i])
    elif int(str) == 3:
        #ckplus_all_path = np.load('app_cnn_6/train_dataset/all_path_2_dataset.npy')
        #Y3 = np.load('app_cnn_6/train_dataset/all_Y_3_dataset.npy')
        for i in range(0, len(filepath[0])):
            print(filepath[0][i])
            set_keras_backend('theano')
            K.set_image_dim_ordering('th')
            neutral_image = auto_main.get_neutral_image(filepath[0][i], automodel)
            set_keras_backend('tensorflow')
            K.set_image_dim_ordering('tf')
            tmp = geo.extract_geometry(filepath[0][i], neutral_image, False)
            if tmp is not -1:
                X_geo.append(tmp)
                Y_geo.append(Y[0][i])

    print('==finish loading geometric data==')
    print('load geo X',len(X_geo))
    return X_geo, Y_geo

# ex) HA, SA dataload
def loading_data_pair(X, Y, emo1, emo2):
    X_pair = []
    Y_pair = []
    for i in range(len(Y)):
        if Y[i] == int(emo1) or Y[i] == int(emo2):
            X_pair.append(X[i])
            Y_pair.append(Y[i])
    print('==loading {0},{1} dataset=='.format((emo1), (emo2)))
    print('X_PAIR.SHAPE:', np.array(X_pair).shape)

    return X_pair, Y_pair


# training data
def CNN(X_pair, Y_pair, X_pair_test, Y_pair_test, emo1, emo2, cycle):
    N = len(Y_pair)
    print("Y pair size : ", N )
    N_test = len(X_pair_test)
    nb_classes = 2
    # process the data to fit in a keras CNN properly
    # input data needs to be (N, C, X, Y) - shaped where
    # N - number of samples
    # C - number of channels per sample
    # (X, Y) - sample size

    X = np.array(X_pair)
    print(X.shape)
    X = X.reshape(N, 36, 1, 1)
    x_test = np.array(X_pair_test).reshape(N_test, 36, 1, 1)

    for i in range(len(Y_pair)):
        if str(Y_pair[i]) == str(emo1):
            Y_pair[i] = 0
        elif str(Y_pair[i]) == str(emo2):
            Y_pair[i] = 1
        else:
            print('not this pair')
    for i in range(len(X_pair_test)):
        if str(Y_pair_test[i]) == str(emo1):
            Y_pair_test[i] = 0
        elif str(Y_pair_test[i]) == str(emo2):
            Y_pair_test[i] = 1
        else:
            print('not this pair')

    y = np_utils.to_categorical(Y_pair, nb_classes)
    y_test = np_utils.to_categorical(Y_pair_test, nb_classes)
    # output labels should be one-hot vectors - ie,
    # 0 -> [0, 0, 1]
    # 1 -> [0, 1, 0]
    # 2 -> [1, 0, 0]
    # define optimizer and objective, compile cnn
    print('final shape:',X.shape)
    cnn = shallow_CNN()
    cnn.compile(loss="categorical_crossentropy", optimizer="adam", metrics=['accuracy'])
    # checkpoint
    filepath = "geo_cnn_6/affect_{0}_fold_{1}_{2}_pair_cnn_model".format((cycle),(emo1), (
    emo2)) + "-weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=False, mode='max', period=5)
    callbacks_list = [checkpoint]
    # train
    # cnn.fit(X, y, epochs=1, callbacks=callbacks_list, verbose=1)
    # cnn.fit(X, y, epochs=1, verbose=1)
    cnn.fit(X, y, validation_data=(x_test,y_test), epochs=30, callbacks=callbacks_list, verbose=1)
    print('==finish training==')
    # cnn.save("geo_weight/{0}_{1}_pair_cnn_model.h5".format((emo1),(emo2)))

    # score = cnn.evaluate(x_test, y_test, verbose=0)
    # print('Test loss:', score[0])


# print('Test accuracy:', score[1])


def train_save(str):
    X = []
    Y = []
    X_geo = []
    Y_geo = []
    # emo1=3
    # emo2=6
    comb_list = list(combinations('0123456', 2))
    # print('comb_list:',comb_list)
    for cycle in range(1,11):
        X, Y = loading_data(str, cycle)

        X_test, Y_test = loading_data_test(str, cycle)
        X_geo, Y_geo = loading_geo_data(X, Y, str)
        
        #geo save
        if not os.path.exists('app_cnn_6/affect/'):
            os.mkdir('app_cnn_6/affect/')
        affect_file1 = 'app_cnn_6/affect/{0}_X_geo.npy'.format((cycle))
        np.save(affect_file1, np.array(X_geo))
        affect_file2 = 'app_cnn_6/affect/{0}_Y_geo.npy'.format((cycle))
        np.save(affect_file2, np.array(Y_geo))

        X_geo_test, Y_geo_test = loading_geo_data(X_test, Y_test, str)
        affect_file3 = 'app_cnn_6/affect/{0}_X_geo_test.npy'.format((cycle))
        np.save(affect_file3, np.array(X_geo_test))
        affect_file4 = 'app_cnn_6/affect/{0}_Y_geo_test.npy'.format((cycle))
        np.save(affect_file4, np.array(Y_geo_test))
        for i in range(len(comb_list)):
            # print('pair:',comb_list[i][0],'+', comb_list[i][1])
            if os.path.exists("geo_cnn_6/affect_{0}_fold_{1}_{2}_pair_cnn_model".format((cycle),(comb_list[i][0]), (comb_list[i][1])) + "-weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"):
                print("already processed...")
                continue
            else:
                X_pair = []
                Y_pair = []
                X_pair, Y_pair = loading_data_pair(X_geo, Y_geo, comb_list[i][0], comb_list[i][1])

                X_pair_test, Y_pair_test = loading_data_pair(X_geo_test, Y_geo_test, comb_list[i][0], comb_list[i][1])

                if len(Y_pair) != 0:
                    CNN(X_pair, Y_pair, X_pair_test, Y_pair_test, comb_list[i][0], comb_list[i][1],cycle)

    # 1by1
    '''X,Y=loading_geo_data()
    X_pair, Y_pair = loading_data_pair(X, Y, 0, 1)
    if len(Y_pair) != 0:
        CNN(X_pair, Y_pair, 0, 1)'''


def test(filepath, emo1, emo2):
    X = []
    Y = []
    geo_list = []
    if filepath.find('ck') is not -1:
        Y_list, geo_list = test_ck_100(emo1, emo2)
        geo_list = np.array(geo_list)
        print('geo_list length: ', len(geo_list), geo_list.shape)
        geo_list = geo_list.reshape((-1, 1, 36, 1))

        model = shallow_CNN("geo_weight/0_1_pair_cnn_model-weights-improvement-00-0.99.hdf5")
        # sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
        # model.compile(optimizer=sgd, loss='categorical_crossentropy')
        count_true = 0
        out = model.predict_classes(geo_list)

        for i in range(len(geo_list)):
            if str(out[i]) == str(0):
                out[i] = emo1
            else:
                out[i] = emo2
            print('True : ' + str(Y_list[i]) + ', Predict : ' + str(out[i]))
            if str(Y_list[i]) == str(out[i]):
                count_true += 1
        print('accuracy: ', (count_true / len(Y_list)) * 100)

        # print(np.argmax(out))

    elif filepath.find('jaffe') is not -1:
        X = []
        Y = []
        X, Y = getdb.get_jaffe_path(X, Y)

        X, Y = select_test_value(X, Y)
        N = len(Y)
        X = np.array(X).reshape((-1, 1, 36, 1))
        geo_fit = getdb.get_jaffe_geo(filepath, X, Y)

        model = shallow_CNN("{0}_{1}_pair_cnn_model.h5".format((emo1), (emo2)))
        # sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
        # model.compile(optimizer=sgd, loss='categorical_crossentropy')
        out = model.predict_classes(geo_fit)
        print('True : ' + str(jaffe.get_emo_int(filepath)) + ', Predict : ' + str(out))
        count_true = 0
        for i in range(len(X)):
            out = model.predict_classes(geo_fit)
            print('True : ' + str(Y[i]) + ', Predict : ' + str(out))
            if str(Y[i]) == str(out):
                count_true += 1
        print('accuracy: ', (count_true / len(X)) * 100)

    elif filepath.find('FERG') is not -1:
        X = []
        Y = []
        X, Y = getdb.get_ferg_path(X, Y)
        X, Y = select_test_value(X, Y)
        N = len(Y)
        X = X.reshape((N, 1, 36, 1))
        for n in range(0, 7):
            if filepath.find(ferg.emo[n]) is not -1:
                emo_true = n
        geo_fit = getdb.get_ferg_geo(filepath, X, Y)

        model = shallow_CNN("{0}_{1}_pair_cnn_model.h5".format((emo1), (emo2)))
        # sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
        # model.compile(optimizer=sgd, loss='categorical_crossentropy')
        count_true = 0
        for i in range(len(X)):
            out = model.predict_classes(geo_fit)
            print('True : ' + str(Y[i]) + ', Predict : ' + str(out))
            if str(Y[i]) == str(out):
                count_true += 1
        print('accuracy: ', (count_true / len(X)) * 100)
    else:
        print('not exist file')




if __name__ == '__main__':
    # train
    X=[]
    Y=[]
    X_pair=[]
    Y_pair=[]
    emo1=3
    emo2=6
    X,Y=loading_geo_data()
    X_pair, Y_pair=loading_data_pair(X, Y, emo1, emo2)
    CNN(X_pair, Y_pair, emo1, emo2)
    #for cycle in range(1,11):
    #train_save(3)

    # test
    '''X=[]
    Y=[]
    X_pair=[]
    Y_pair=[]
    #emo1=3
    #emo2=6
    comb_list = list(combinations('0123456',2))
    #print('comb_list:',comb_list)
    for i in range(len(comb_list)):
        #X,Y=loading_geo_data()
        #print('pair:',comb_list[i][0],'+', comb_list[i][1])
        if os.path.exists("geo_model/{0}_{1}_pair_cnn_model.h5".format((comb_list[i][0]),(comb_list[i][1]))):
            test('ck', comb_list[i][0], comb_list[i][1])'''

    # 1by1 test
    # test('ck', 0, 1)
