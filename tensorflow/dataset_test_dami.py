# -*- coding: utf-8 -*-
"""
Created on Tue Sep  3 20:42:34 2019

@author: IVPL-SERVER
"""

import numpy as np 

X = np.load('app_cnn_6/train_dataset/all_path_2_dataset.npy')
Y = np.load('app_cnn_6/train_dataset/all_Y_2_dataset.npy')

print(X)
for i in range(len(X)):
    print(X[i])