# -*- coding: utf-8 -*-
"""
Created on Sat Feb 24 18:44:41 2018

@author: KIMJIHAE
"""
import convert_ckplus_files as ck
import convert_jaffe_files as jaffe
import convert_ferg_files as ferg
import geometric as geo
import os
from os import listdir
from os.path import isfile, join
import csv
import pandas as pd 
import numpy as np
import keras.backend as K
from os import environ
from importlib import reload
def set_keras_backend(backend):
    if K.backend() != backend:
        environ['KERAS_BACKEND'] = backend
        reload(K)
        assert K.backend() == backend
set_keras_backend('tensorflow')
K.set_image_dim_ordering('tf')
dirname_ck = 'ckplus/extended-cohn-kanade-images/cohn-kanade-images/'
def get_ck_plus_path(dirname_ck,X,Y):
    try:
        filenames = listdir(dirname_ck)
        for filename in filenames:
            
            full_filename = os.path.join(dirname_ck, filename)
            emo = ck.get_emo_int(full_filename)
            if os.path.isdir(full_filename):
                #convert_ckplus_files(full_filename)
                get_ck_plus_path(full_filename,X,Y)
                #print('full_filename:',full_filename)
            else:
                #ext = os.path.splitext(full_filename)[-1]
                ext = os.path.basename(full_filename)
                ext = os.path.splitext(ext)[-1]
                resizetxt = os.path.basename(full_filename)[0:6]
                
                if ext == '.png' and resizetxt == 'Resize' and emo != 8:
                    X.append(full_filename)
                    Y.append(ck.get_emo_int(full_filename))
        return X,Y
    except PermissionError:
        pass    

def get_ck_plus_geo(filepath, X, Y):
    diff_list=[]
    for i in range(len(X)):
        if os.path.basename(X[i])[:15]==os.path.basename(filepath)[:15] and Y[i]==6:
            #print(os.path.basename(X[i])[:15])
            diff_g_vec = geo.extract_geometry(filepath, X[i]) 
            diff_list.append(diff_g_vec)
    
    return diff_list

def make_list_ck_plus_geo(X_geo,Y_geo):
    X=[]
    Y=[]
    X, Y = get_ck_plus_path(dirname_ck,X,Y)
    for i in range(len(X)):   
        tmp = get_ck_plus_geo(X[i],X,Y)
        if tmp is not -1:
            for j in range(len(tmp)):
                X_geo.append(tmp[j])
                Y_geo.append(Y[i])
                print('X:',X[i],'+',j,' 번째')
    return X_geo,Y_geo            
        
def get_jaffe_path(X,Y):
    dirname = 'jaffe/'
    file_list = listdir(dirname)
    for files in file_list:
        if files.find('change') is not -1:
            full_filename = os.path.join(dirname, files)
            
            X.append(full_filename)
            Y.append(jaffe.get_emo_int(files))
    return X, Y

def get_jaffe_geo(filepath,X,Y):
    diff_list=[]
    for i in range(len(X)):
        if os.path.basename(X[i])[:2]==os.path.basename(filepath)[:2] and Y[i]==6:
            #print(os.path.basename(X[i])[:2])
            diff_g_vec = geo.extract_geometry(filepath, X[i]) 
            diff_list.append(diff_g_vec)
            break
            #print('get_jaffe_geo:',diff_g_vec)
    return diff_list

def make_list_jaffe_geo(X_geo,Y_geo):
    X=[]
    Y=[]
    tmp=[]
    X, Y = get_jaffe_path(X,Y)
    for i in range(len(X)):   
        tmp = get_jaffe_geo(X[i],X,Y)
        if tmp is not -1:
            for j in range(len(tmp)):
                X_geo.append(tmp[j])
                Y_geo.append(Y[i])
                print('X:',X[i],'+',j,' 번째')        
    return X_geo,Y_geo

def get_ferg_path(X,Y):
    emo = ['anger','disgust','fear','joy','sadness','surprise','neutral']
    names = ['aia','bonnie','jules','malcolm','mery','ray']
    print('--------get_ferg_path함수 진입--------')
    for n in range(0,6):
        for a in range(0,7):
            file_dir = ferg.folder_access(emo[a],names[n])
            if file_dir=='':
                pass
            else:
                file_list = listdir(file_dir)
                for files in file_list:
                    full_filename = os.path.join(file_dir, files)
                    if files.find('Resize_') is not -1:
                        X.append(full_filename)
                        #print('path:',full_filename)
                        Y.append(a)
                        #print('label:',a)
    print('---------get_ferg_path 완료--------------\n')
    return X, Y

def get_ferg_geo(filepath,X,Y):
    diff_list=[]
    print('--------get_ferg_geo함수 진입--------')
    for i in range(len(X)):
        dir1, dir2 = os.path.split(os.path.dirname(filepath))
        dir3, name = os.path.split(dir1)
        
        dir4, dir5 = os.path.split(os.path.dirname(X[i]))
        dir6, search = os.path.split(dir4)
        
        if search==name and Y[i]==6:
            print('get_geo file:',filepath,'+',i,"번째")
            #print(X[i])
            diff_g_vec = geo.extract_geometry(filepath, X[i]) 
            diff_list.append(diff_g_vec)
            break
    print('---------get_ferg_geo 완료--------------\n')
    return diff_list

def make_list_ferg_geo(X_geo,Y_geo):
    X=[]
    Y=[]
    X, Y = get_ferg_path(X,Y)
    for i in range(len(X)):   
        tmp = get_ferg_geo(X[i],X,Y)
        if tmp is not -1:
            for j in range(len(tmp)):
                X_geo.append(tmp[j])
                Y_geo.append(Y[i])
        
    return X_geo,Y_geo


def all_path():
    X_ck = []
    Y_ck = []
    X_jaffe = []
    Y_jaffe = []
    X_ferg = []
    Y_ferg = []
    
    X_jaffe, Y_jaffe = make_list_jaffe_geo(X_jaffe,Y_jaffe)
    df1 = pd.DataFrame(X_jaffe)
    df1.to_csv("geo_vec_jaffe.csv",header=False, index=False)

    df2 = pd.DataFrame(Y_jaffe)
    df2.to_csv("geo_label_jaffe.csv",header=False, index=False)
    
    print('jaffe dataset length:',len(X_jaffe))
        
    X_ck, Y_ck = make_list_ck_plus_geo(X_ck,Y_ck)
    df1 = pd.DataFrame(X_ck)
    df1.to_csv('geo_vec_ck.csv', header=False, index=False)
    
    df2 = pd.DataFrame(Y_ck)
    df2.to_csv('geo_label_ck.csv', header=False, index=False)

    print('ck plus dataset length:',len(X_ck)) 

    X=[]
    Y=[]
    X, Y = get_ferg_path(X,Y)
    for i in range(len(X)): 
        tmp=[]
        try:
            tmp = get_ferg_geo(X[i],X,Y)
            
            #print('label:',Y[i])
            #Y_geo2 = np.array(Y_geo2)
            #Y_geo2 = pd.Series(Y_geo2)
        except OSError as e:
            print('----------os error')
            continue
        
        for j in range(len(tmp)):
            if tmp[j] is not -1:
                print('---------geo vec 저장시작---------')
                X_ferg = tmp[j]
                #df1 = pd.read_csv('geo_vec.csv')
                #print('ferg x_geo:',X_geo2)
                df1_new = pd.DataFrame(X_ferg)
                df1_new = df1_new.T
                #df1.append(df1_new, ignore_index=True)
                df1_new.to_csv('geo_vec_ferg.csv', mode='a', header=False, index=False)
                #pd.read_csv('geo_vec.csv').append(df1).to_csv('geo_vec.csv')
                print('---------geo vec 저장완료---------\n')

                print('---------geo vec 저장시작---------')
                Y_ferg = Y[i]
                #print(Y_geo2)
                #Y_geo2 = np.array(Y_geo2)
                #df2 = pd.read_csv('geo_label.csv')
                #print('ferg y_geo:',Y_geo2)
                f = open('geo_label_ferg.csv','a',newline='')
                writer=csv.writer(f)
                writer.writerow([Y_ferg])
                f.close()
                #df2_new = pd.DataFrame(Y_geo2)
                #df2.append(df2_new, ignore_index=True)
                #df2_new.to_csv('geo_label.csv', mode='a',header=False, index=False)
                print('X:',X[i],'+',j,' 번째')
                print('---------geo vec 저장완료---------\n')
'''    X_geo = []
    Y_geo = []
    X_geo1 = []
    Y_geo1 = []
    X_geo2 = []
    Y_geo2 = []    
    
    X_geo, Y_geo = make_list_jaffe_geo(X_geo,Y_geo)
    df1 = pd.DataFrame(X_geo)
    df1.to_csv("geo_vec.csv",header=False, index=False)

    df2 = pd.DataFrame(Y_geo)
    df2.to_csv("geo_label.csv",header=False, index=False)
    
        
    X_geo1, Y_geo1 = make_list_ck_plus_geo(X_geo1,Y_geo1)

    df1 = pd.read_csv('geo_vec.csv')
    df1_new = pd.DataFrame(X_geo1)
    df1.append(df1_new, ignore_index=True)
    df1.to_csv('geo_vec.csv', mode='a',header=False, index=False)
    #pd.read_csv('geo_vec.csv').append(df1).to_csv('geo_vec.csv')
    
    df2 = pd.read_csv('geo_label.csv')
    df2_new = pd.DataFrame(Y_geo1)
    df2.append(df2_new, ignore_index=True)
    df2.to_csv('geo_label.csv', mode='a',header=False, index=False)
    #pd.read_csv('geo_vec.csv').append(df1).to_csv('geo_vec.csv')
    
    X=[]
    Y=[]
    X, Y = get_ferg_path(X,Y)
    for i in range(len(X)): 
        tmp=[]
        try:
            tmp = get_ferg_geo(X[i],X,Y)
            
            #print('label:',Y[i])
            #Y_geo2 = np.array(Y_geo2)
            #Y_geo2 = pd.Series(Y_geo2)
        except OSError as e:
            print('----------os error')
            continue
        
        for j in range(len(tmp)):
            if tmp[j] is not -1:
                print('---------geo vec 저장시작---------')
                X_geo2 = tmp[j]
                #df1 = pd.read_csv('geo_vec.csv')
                #print('ferg x_geo:',X_geo2)
                df1_new = pd.DataFrame(X_geo2)
                #df1.append(df1_new, ignore_index=True)
                df1_new.to_csv('geo_vec.csv', mode='a',header=False, index=False)
                #pd.read_csv('geo_vec.csv').append(df1).to_csv('geo_vec.csv')
                print('---------geo vec 저장완료---------\n')

                print('---------geo vec 저장시작---------')
                Y_geo2 = Y[i]
                #print(Y_geo2)
                #Y_geo2 = np.array(Y_geo2)
                #df2 = pd.read_csv('geo_label.csv')
                #print('ferg y_geo:',Y_geo2)
                with open('geo_label.csv','a') as f:
                    writer=csv.writer(f)
                    writer.writerow([Y_geo2])
                #df2_new = pd.DataFrame(Y_geo2)
                #df2.append(df2_new, ignore_index=True)
                #df2_new.to_csv('geo_label.csv', mode='a',header=False, index=False)
                print('X:',X[i],'+',j,' 번째')
                print('---------geo vec 저장완료---------\n')
'''                    
    
if __name__ == '__main__': 
    all_path()