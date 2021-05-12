# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 10:46:07 2018

@author: KIMJIHAE
"""

#FERG이미지(png 256*256)를 128*128로 변형하고 다시 저장하는 과정.
import os
from os import listdir
from os.path import isfile, join
import re
from PIL import Image
import numpy as np
import lbp as lbp
import cv2
import random
dir = 'FERG_DB_256/'
#EMOS = {"AN": 0, "DI": 1, "FE": 2, "HA": 3, "SA": 4, "SU": 5,"NE": 6}
#EMOS = {"anger": 0, "disgust": 1, "fear": 2, "joy": 3, "sadness": 4, "surprise": 5,"neutral": 6}
emo = ['anger','disgust','fear','joy','sadness','surprise','neutral']
names = ['aia','bonnie','jules','malcolm','mery','ray']
#convert tiff to png
def convert_ferg_files():
    for n in range(0,6):
        for a in range(0,7):
            file_dir = folder_access(emo[a],names[n])
            #print(str(n)+' 사람'+str(a)+'감정')
            onlyfiles = [join(file_dir,f) for f in listdir(file_dir) if isfile(join(file_dir, f))]
            png_files = [f for f in onlyfiles if f.endswith(".png")]
            for file_name in png_files:
                if file_name.find('Resize_') is -1:
                    #full_filename = os.path.join(file_dir, file_name)
                    #print('dir: '+full_filename)
                    im = Image.open(file_name)
                    im = im.resize((128, 128))
                    title, ext = os.path.splitext(os.path.basename(file_name))
                    newname = title.replace(names[n],'Resize_'+names[n])
                    im.convert("L").save(os.path.join(file_dir,newname+ext))

def folder_access(emo,name):
    dir = 'FERG_DB_256/'
    file_dir=''
    if os.path.exists(dir+name+'/'):
        folder_list = os.listdir(dir+name+'/')
        for folders in folder_list:
            if folders.find(emo) is not -1:
                file_dir = dir+name+'/'+folders+'/'
        #print(file_dir)
    return file_dir

def getDataFERG():
    #convert_ferg_files()  
    Y = []
    X = []
    #first = True
    for n in range(0,6):
        for a in range(0,7):
            file_dir = folder_access(emo[a],names[n])
            file_list = listdir(file_dir)
            for files in file_list:
                full_filename = os.path.join(file_dir, files)
                pixel=[]
                #if first:
                #    first = False
                #else:
                if files.find('Resize_') is not -1:
                    im = Image.open(full_filename)
                    img_gray = np.array(im).reshape(128,128)
                    X.append(img_gray)
                    Y.append(a)

    X, Y = np.array(X) / 255.0, np.array(Y)
    #print('getDataJaffe X.shape',X.shape)
    #print('이미지갯수:',len(X),len(Y))
    return X, Y

def getDataFERG_LBP(count,batch):
    #convert_ferg_files()  
    Y = []
    X = []
    file_path=[]
    for n in range(0,6):
        for a in range(0,7):
            file_dir = folder_access(emo[a],names[n])
            file_list = listdir(file_dir)
            new_file_lists=[]
            for files in file_list:
                if files.find('Resize_') is not -1:
                    new_file_lists.append(files)
            batch_start = 0 + (batch * int(count/6))
            batch_end = batch_start + int(count/6)
            file_list_batch = new_file_lists[batch_start:batch_end]
            for files in file_list_batch:
                full_filename = os.path.join(file_dir, files)
                #im = Image.open(full_filename)
                img_bgr = cv2.imread(full_filename)
                height, width, channel = img_bgr.shape
                img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

                img_lbp = np.zeros((height, width), np.uint8)
                for i in range(0, 128):
                    for j in range(0, 128):
                         img_lbp[i, j] = lbp.lbp_calculated_pixel(img_gray, i, j) #각 픽셀마다 lbp 계산해서 배열에 입력.

                X.append(img_lbp)
                Y.append(a)
                file_path.append(full_filename)

    X, Y, file_path = np.array(X) / 255.0, np.array(Y), np.array(file_path)

    return X, Y, file_path


def getImageDataFERG():#cnn main에서 불러오는 부분...
    X, Y = getDataFERG()
    N, D = X.shape
    #D, N = X.shape
    #print('getImageDataJaffe X.shape',X.shape)
    d = int(np.sqrt(D))
    X = X.reshape(N, 1, d, d)
    return X, Y

def getImageDataFERG_LBP(count,batch):
    X, Y, file_list = getDataFERG_LBP(count,batch)
    print('X.SHAPE:',X.shape)
    N, D1, D2 = X.shape
    #D, N = X.shape
    #print('getImageDataJaffe X.shape',X.shape)
    #d = int(np.sqrt(D))
    X = X.reshape(N, 1, D1, D2)
    return X, Y, file_list
if __name__ == '__main__':
    #convert_jaffe_files(id, dir)
    getDataFERG_LBP()
       