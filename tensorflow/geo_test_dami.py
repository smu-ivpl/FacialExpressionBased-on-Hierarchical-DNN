# -*- coding: utf-8 -*-
"""
Created on Tue Sep  3 19:55:00 2019

@author: IVPL-SERVER
"""


import numpy as np
from geo_cnn_keras import shallow_CNN
import geometric as geo
from keras import backend as K
from os import environ
from importlib import reload
import os


def set_keras_backend(backend):
    if K.backend() != backend:
        environ['KERAS_BACKEND'] = backend
        reload(K)
        assert K.backend() == backend


set_keras_backend('tensorflow')
K.set_image_dim_ordering('tf')

filepath = "x.npy"

def image_load(cycle):
    X = np.load('app_cnn_6/test_dataset/{0}/{1}_LBP_X_test_dataset.npy'.format((cycle), (2)))
    Y = np.load('app_cnn_6/test_dataset/{0}/{1}_Y_test_dataset.npy'.format((cycle), (2)))
    print('X SHAPE:',X.shape)
    print('Y SHAPE:',Y.shape)

    #X, Y = shuffle(X, Y)
    print(len(X), len(Y))
    print('--->     Loading ck+ Image Data      <---')
    return X, Y

def image_load_geo(str,file_path_idx):
    dirname_ck = 'ckplus/extended-cohn-kanade-images/cohn-kanade-images/'
    X = []
    Y = []
    X = np.load('app_cnn_6/train_dataset/all_path_2_dataset.npy')
    Y = np.load('app_cnn_6/train_dataset/all_Y_2_dataset.npy')

    for i in range(len(X)):
        if os.path.basename(X[i])[:15] == os.path.basename(X[file_path_idx])[:15] and Y[i] == 6:
            print(X[i], Y[i])
            print("!!!", os.path.basename(X[file_path_idx])[:15])
            print("----:", Y[i])
            diff_g_vec = geo.extract_geometry(X[file_path_idx], X[i])
            break
    return diff_g_vec

def geometric_cnn(file_path_idx, cycle):
    print("====", file_path_idx)
    #print('emo1, emo2: ', emo_1, '+', emo_2)

    '''
    list_of_files = glob.glob('geo_cnn_6/ck_{0}_fold_{1}_{2}_pair_cnn_model'.format((cycle), (emo_1), (emo_2)) + "*.hdf5")
    filepath = max(list_of_files, key=os.path.getctime)
    print("loading:", filepath)

    if os.path.exists(filepath):'''
    model = shallow_CNN()
    geo_vec = []
    geo_vec = image_load_geo(str, file_path_idx) #differences
    if geo_vec is not -1:
        geo_vec = geo_vec.reshape(-1, 36, 1, 1)

        get_softmax_result_geo = model.predict(geo_vec)
        return get_softmax_result_geo[0]
    else:
        return -1



if __name__ == '__main__':
    X = []
    Y = []
    weights = [0.8]
    X, Y = image_load(1)
    for i in range(0,1):
        #geometric_result_list = geometric_cnn(file_list[0][i], 1)
        geometric_result_list = geometric_cnn(i, 1)
    print(geometric_result_list)
    K.clear_session()