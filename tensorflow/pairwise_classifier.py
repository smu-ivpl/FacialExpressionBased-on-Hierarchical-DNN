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


#loading data
def loading_geo_data():
    # images are 48x48 = 2304 size vectors
    # N = 35887
    Y1 = []
    X1 = []
    Y2 = []
    X2 = []
    Y3 = []
    X3 = []
    readfile = open('geo_feat/geo_label_ck.csv', "r")
    readlines = readfile.readlines()
    for line in readlines:
        row = line.split(',')
        Y1.append(int(row[0]))
    count1 = []
    count1 = count_dataset(Y1)
    print('>CK+ Dataset')
    print('Angry:    ',count1[0])
    print('Disgust:  ',count1[1])
    print('Fear:     ',count1[2])
    print('Happy:    ',count1[3])
    print('Sad:      ',count1[4])
    print('Surprise: ',count1[5])
    print('Neutral:  ',count1[6])
    print('total:    ',sum(count1))
    
    
    readfile = open('geo_feat/geo_label_jaffe.csv', "r")
    readlines = readfile.readlines()
    for line in readlines:
        row = line.split(',')
        Y2.append(int(row[0]))
    count2 = []
    count2 = count_dataset(Y2)
    print('>JAFFE Dataset')
    print('Angry:    ',count2[0])
    print('Disgust:  ',count2[1])
    print('Fear:     ',count2[2])
    print('Happy:    ',count2[3])
    print('Sad:      ',count2[4])
    print('Surprise: ',count2[5])
    print('Neutral:  ',count2[6])
    print('total:    ',sum(count2))

    readfile = open('geo_feat/geo_label_ferg.csv', "r")
    readlines = readfile.readlines()
    for line in readlines:
        row = line.split(',')
        Y3.append(int(row[0]))
    count3 = []
    count3 = count_dataset(Y3)
    print('>FERG Dataset')
    print('Angry:    ',count3[0])
    print('Disgust:  ',count3[1])
    print('Fear:     ',count3[2])
    print('Happy:    ',count3[3])
    print('Sad:      ',count3[4])
    print('Surprise: ',count3[5])
    print('Neutral:  ',count3[6])
    print('total:    ',sum(count3))
    
    readfile = open('geo_feat/geo_vec_ck.csv', "r")
    readlines = readfile.readlines()
    for line in readlines:
        X1.append(line.strip().split(','))
    print("==finish loading ck vectors==")
    
    readfile = open('geo_feat/geo_vec_jaffe.csv', "r")
    readlines = readfile.readlines()
    for line in readlines:        
        X2.append(line.strip().split(','))
    print("==finish loading jaffe vectors==")
    
    readfile = open('geo_feat/geo_vec_ferg.csv', "r")
    readlines = readfile.readlines()
    for line in readlines:
        X3.append(line.strip().split(','))
    print("==finish loading ferg vectors==")

    X2 = np.concatenate((X2, X2), axis=0)
    Y2 = np.concatenate((Y2, Y2), axis=0)
    X2, Y2 = loading_n_data(X2, Y2, count2)
    X3, Y3 = get_n_dataset(X3,Y3)
    X=np.concatenate((X2, X3), axis=0)
    X=np.concatenate((X, X2), axis=0)
    X=np.concatenate((X, X1), axis=0)

    Y=np.concatenate((Y2, Y3), axis=0)
    Y=np.concatenate((Y, Y2), axis=0)
    Y=np.concatenate((Y, Y1), axis=0)

    X, Y = shuffle(X, Y)
    print(len(X),len(Y))
    print('==finish loading geometric data==')
    return X, Y

# ex) HA, SA dataload
def loading_data_pair(X, Y, emo1, emo2):
    X_pair=[]
    Y_pair=[]
    for i in range(len(Y)):
        if Y[i]==int(emo1) or Y[i]==int(emo2):
            X_pair.append(X[i])
            Y_pair.append(Y[i])
    print('==loading {0},{1} dataset=='.format((emo1),(emo2)))
    print('X_PAIR.SHAPE:',len(X_pair[0]))
    
    return X_pair, Y_pair

# training data
def CNN(X_pair, Y_pair, emo1, emo2):
    N = len(Y_pair)
    nb_classes=2    
    # process the data to fit in a keras CNN properly
    # input data needs to be (N, C, X, Y) - shaped where
    # N - number of samples
    # C - number of channels per sample
    # (X, Y) - sample size
    
    X = np.array(X_pair)
    print(X.shape)
    X = X.reshape((N, 1, 36, 1))
    x_test = np.array(X_pair).reshape((N, 1, 36, 1))
    
    for i in range(len(Y_pair)):
        if str(Y_pair[i])==str(emo1):
            Y_pair[i] = 0
        elif str(Y_pair[i])==str(emo2):
            Y_pair[i] = 1
        else:
            print('not this pair')
            
    y = np_utils.to_categorical(Y_pair, nb_classes)
    y_test = np_utils.to_categorical(Y_pair, nb_classes)
    # output labels should be one-hot vectors - ie,
    # 0 -> [0, 0, 1]
    # 1 -> [0, 1, 0]
    # 2 -> [1, 0, 0]
    # define optimizer and objective, compile cnn
    cnn = shallow_CNN()
    cnn.compile(loss="categorical_crossentropy", optimizer="adam", metrics=['accuracy'])
    # checkpoint
    filepath="geo_weight/{0}_{1}_pair_cnn_model".format((emo1),(emo2))+"-weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=False, mode='max', period=5)
    callbacks_list = [checkpoint]
    # train    
    #cnn.fit(X, y, epochs=1, callbacks=callbacks_list, verbose=1)
    #cnn.fit(X, y, epochs=1, verbose=1)
    cnn.fit(X, y, validation_split=0.33, epochs=30, callbacks=callbacks_list, verbose=1)
    print('==finish training==')
    score = cnn.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    #cnn.save("geo_weight/{0}_{1}_pair_cnn_model.h5".format((emo1),(emo2)))
    
    #score = cnn.evaluate(x_test, y_test, verbose=0)
    #print('Test loss:', score[0])
   #print('Test accuracy:', score[1])
    

def train_save():
    X=[]
    Y=[]
    X_pair=[]
    Y_pair=[]
    #emo1=3
    #emo2=6
    comb_list = list(combinations('0123456',2))
    #print('comb_list:',comb_list)
    for i in range(len(comb_list)):
        X,Y=loading_geo_data()
        #print('pair:',comb_list[i][0],'+', comb_list[i][1])
        if os.path.exists("geo_weight/{0}_{1}_pair_cnn_model".format((comb_list[i][0]),(comb_list[i][1]))+"-weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"):
            print("already processed...")
            continue
        else:
            X_pair, Y_pair = loading_data_pair(X, Y, comb_list[i][0], comb_list[i][1])
            if len(Y_pair) != 0:
                CNN(X_pair, Y_pair, comb_list[i][0], comb_list[i][1])
                
    #1by1
    '''X,Y=loading_geo_data()
    X_pair, Y_pair = loading_data_pair(X, Y, 0, 1)
    if len(Y_pair) != 0:
        CNN(X_pair, Y_pair, 0, 1)'''

def test(filepath, emo1, emo2):
    X=[]
    Y=[]
    geo_list=[]
    if filepath.find('ck') is not -1:
        Y_list,geo_list = test_ck_100(emo1,emo2)
        geo_list=np.array(geo_list)
        print('geo_list length: ',len(geo_list),geo_list.shape)
        geo_list = geo_list.reshape((-1, 1, 36, 1))
        
        model = shallow_CNN("geo_weight/0_1_pair_cnn_model-weights-improvement-00-0.99.hdf5")
        #sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
        #model.compile(optimizer=sgd, loss='categorical_crossentropy')
        count_true=0
        out = model.predict_classes(geo_list)
        
        for i in range(len(geo_list)):
            if str(out[i])==str(0):
                out[i]=emo1
            else:
                out[i]=emo2
            print('True : ' + str(Y_list[i]) + ', Predict : ' + str(out[i]))
            if str(Y_list[i]) == str(out[i]):
                count_true += 1
        print('accuracy: ',(count_true/len(Y_list))*100)   
        
        
        #print(np.argmax(out))
        
    elif filepath.find('jaffe') is not -1:
        X=[]
        Y=[]
        X, Y = getdb.get_jaffe_path(X,Y)
        
        X, Y=select_test_value(X,Y)
        N = len(Y)
        X = np.array(X).reshape((-1, 1, 36, 1))
        geo_fit = getdb.get_jaffe_geo(filepath, X, Y)
        
        model = shallow_CNN("{0}_{1}_pair_cnn_model.h5".format((emo1),(emo2)))
        #sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
        #model.compile(optimizer=sgd, loss='categorical_crossentropy')
        out = model.predict_classes(geo_fit)
        print('True : ' + str(jaffe.get_emo_int(filepath)) + ', Predict : ' + str(out))
        count_true=0
        for i in range(len(X)):
            out = model.predict_classes(geo_fit)
            print('True : ' + str(Y[i]) + ', Predict : ' + str(out))
            if str(Y[i]) == str(out):
                count_true += 1
        print('accuracy: ',(count_true/len(X))*100)
        
    elif filepath.find('FERG') is not -1:
        X=[]
        Y=[]
        X, Y = getdb.get_ferg_path(X,Y)
        X, Y=select_test_value(X,Y)
        N = len(Y)
        X = X.reshape((N, 1, 36, 1))
        for n in range (0,7):
            if filepath.find(ferg.emo[n]) is not -1:
                emo_true = n
        geo_fit = getdb.get_ferg_geo(filepath, X, Y)
        
        model = shallow_CNN("{0}_{1}_pair_cnn_model.h5".format((emo1),(emo2)))
        #sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
        #model.compile(optimizer=sgd, loss='categorical_crossentropy')
        count_true=0
        for i in range(len(X)):
            out = model.predict_classes(geo_fit)
            print('True : ' + str(Y[i]) + ', Predict : ' + str(out))
            if str(Y[i]) == str(out):
                count_true += 1
        print('accuracy: ',(count_true/len(X))*100)  
    else:
        print('not exist file')

def test_ck_100(emo1, emo2):
    X=[]
    Y=[]
    Y_list=[]
    diff_list=[]
    count_100=0
    X, Y = getdb.get_ck_plus_path(dirname_ck,X,Y)
    for j in range(len(X)):
        if str(Y[j])==str(emo1) or str(Y[j])==str(emo2):
            for i in range(len(X)):
                if Y[i]==6:
                    if os.path.basename(X[i])[:15]==os.path.basename(X[j])[:15] and count_100!=100:
                        #print(os.path.basename(X[i])[:15])
                        diff_g_vec = geo.extract_geometry(X[j], X[i]) 
                        diff_list.append(diff_g_vec)
                        Y_list.append(Y[j])
                        count_100 += 1
    return Y_list,diff_list

def test_jaffe_100(emo1, emo2):
    X=[]
    Y=[]
    diff_list=[]
    count_100=0
    X, Y = getdb.get_jaffe_path(X,Y)
    for j in range(len(X)):
        if Y[j]==emo1 or Y[j]==emo2:
            for i in range(len(X)):
                if os.path.basename(X[i])[:15]==os.path.basename(X[j])[:15] and Y[i]==6 and count_100!=100:
                    #print(os.path.basename(X[i])[:15])
                    diff_g_vec = geo.extract_geometry(X[j], X[i]) 
                    diff_list.append(diff_g_vec)
                    count_100 += 1
    return diff_list

def test_ferg_100(emo1, emo2):
    X=[]
    Y=[]
    diff_list=[]
    count_100=0
    X, Y = getdb.get_ferg_path(X,Y)
    for j in range(len(X)):
        if Y[j]==emo1 or Y[j]==emo2:
            for i in range(len(X)):
                if os.path.basename(X[i])[:15]==os.path.basename(X[j])[:15] and Y[i]==6 and count_100!=100:
                    #print(os.path.basename(X[i])[:15])
                    diff_g_vec = geo.extract_geometry(X[j], X[i]) 
                    diff_list.append(diff_g_vec)
                    count_100 += 1
    return diff_list
    

if __name__ == '__main__': 
    #train
    '''X=[]
    Y=[]
    X_pair=[]
    Y_pair=[]
    emo1=3
    emo2=6
    X,Y=loading_geo_data()
    X_pair, Y_pair=loading_data_pair(X, Y, emo1, emo2)
    CNN(X_pair, Y_pair, emo1, emo2)'''
    train_save()
    
    #test
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
    
    #1by1 test
    #test('ck', 0, 1)
    