# -*- coding: utf-8 -*-
"""
Created on Tue Sep 12 17:11:52 2017

@author: kimjihae
"""

from os import listdir
import dlib
import imutils
import numpy as np
import os
from PIL import Image
from crop_images import faceImageCrop
from decimal import Decimal
import lbp as lbp
import cv2
from util import sp_noise
from sklearn.utils import shuffle
import pandas as pd

EMOS = {"1": 0, "3": 1, "4": 2, "5": 3, "6": 4, "7": 5, "0": 6}


# convert ck+ crop & resize image
# sub dir 탐색하며 png파일 찾아 crop하여 저장. 이름은 Resize_+원래 파일명
def convert_ckplus_files(dirname):
    try:
        filenames = listdir(dirname)
        for filename in filenames:
            full_filename = os.path.join(dirname, filename)
            if os.path.isdir(full_filename):
                convert_ckplus_files(full_filename)
            else:
                ext = os.path.splitext(full_filename)[-1]
                resizetxt = os.path.basename(full_filename)[0:6]
                # print('resizetxt:',resizetxt)
                if ext == '.png':
                    if resizetxt != 'Resize':
                        im = Image.open(full_filename)
                        ###crop
                        cropimg = faceImageCrop(full_filename)
                        ##################################################
                        im = cropimg.resize((128, 128))
                        newname = filename.replace('S', 'Resize_S')
                        # print(newname)
                        im.convert("L").save(os.path.join(dirname, newname))
                    # print(full_filename)
    except PermissionError:
        pass


def delete_ckplus_files(dirname):
    try:
        filenames = listdir(dirname)
        for filename in filenames:

            full_filename = os.path.join(dirname, filename)
            if os.path.isdir(full_filename):
                # convert_ckplus_files(full_filename)
                delete_ckplus_files(full_filename)
                # print('full_filename:',full_filename)
            else:
                # ext = os.path.splitext(full_filename)[-1]
                ext = os.path.basename(full_filename)
                ext = os.path.splitext(ext)[-1]
                resizetxt = os.path.basename(full_filename)[0:6]

                if ext == '.png' and resizetxt == 'Resize':
                    os.remove(full_filename)
    except PermissionError:
        pass


# dir C:\FacialExpression\facialExpression\tensorflow\ckplus\Emotion_labels\Emotion 의 첫번쨰 숫자 emotion
def get_emo_int(full_filename):
    # pieces = filename.split('/')
    # emo_key = filename[0:2]
    # return EMOS[emo_key]
    dirname = os.path.dirname(full_filename)
    filename = os.path.basename(full_filename)

    newfilename = filename.replace('Resize_S', 'S')
    filenamepng = os.path.splitext(newfilename)
    newfilename = filenamepng[0] + '_emotion.txt'

    newdirname = dirname.replace('extended-cohn-kanade-images/cohn-kanade-images', 'Emotion_labels/Emotion')
    newfull_name = os.path.join(newdirname, newfilename)
    if filenamepng[1] == '.png':
        newfull_name1 = os.path.join(newdirname, (
                filenamepng[0][0:14] + str(int(filenamepng[0][14:]) + 1).zfill(3) + '_emotion.txt'))
        newfull_name2 = os.path.join(newdirname, (
                filenamepng[0][0:14] + str(int(filenamepng[0][14:]) + 2).zfill(3) + '_emotion.txt'))
        # newfull_name3 = os.path.join(newdirname,(filenamepng[0][0:14]+str(int(filenamepng[0][14:])+3).zfill(3)+'_emotion.txt'))
        # file read
        if os.path.isfile(newfull_name):
            f = open(newfull_name, 'r')
            line = f.readline()
            em = str(int(Decimal(line)))
            if filename[22:24] == '01':
                return 6
            else:
                if em == '2':
                    return 8
                else:
                    return EMOS[em]

        elif os.path.isfile(newfull_name1):
            f = open(newfull_name1, 'r')
            line = f.readline()
            em = str(int(Decimal(line)))
            if filename[22:24] == '01':
                return 6
            else:
                if em == '2':
                    return 8
                else:
                    return EMOS[em]
        elif os.path.isfile(newfull_name2):
            f = open(newfull_name2, 'r')
            line = f.readline()
            em = str(int(Decimal(line)))
            if filename[22:24] == '01':
                return 6
            else:
                if em == '2':
                    return 8
                else:
                    return EMOS[em]

        else:
            if filename[22:24] == '01':
                return 6
            else:
                return 8
    # dir이름을 rename해서 라벨 있는 폴더로간다.
    # 가서 있으면 첫번째 숫자리턴
    # 없으면 8리턴 (X,Y APPEND하는 부분예써 IF절로 걸러.)


def getXYckplus_load_gray(dirname, X, Y, file_path):
    try:
        filenames = listdir(dirname)
        for filename in filenames:
            full_filename = os.path.join(dirname, filename)
            emo = get_emo_int(full_filename)
            if os.path.isdir(full_filename):
                getXYckplus_load_gray(full_filename, X, Y, file_path)
            else:
                ext = os.path.basename(full_filename)
                ext = os.path.splitext(ext)[-1]
                resizetxt = os.path.basename(full_filename)[0:6]

                if ext == '.png' and resizetxt == 'Resize' and emo != 8:
                    pixel = []
                    global count
                    img_bgr = cv2.imread(full_filename)
                    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
                    img_gray = img_gray.reshape((128, 128))
                    X.append(img_gray)
                    Y.append(get_emo_int(full_filename))
                    file_path.append(full_filename)
        return X, Y, file_path
    except PermissionError:
        pass


def getXYckplus_load_rotation(dirname, X, Y, angle):
    face_detector = dlib.get_frontal_face_detector()
    try:
        filenames = listdir(dirname)
        for filename in filenames:
            full_filename = os.path.join(dirname, filename)
            emo = get_emo_int(full_filename)
            if os.path.isdir(full_filename):
                getXYckplus_load_rotation(full_filename, X, Y, angle)
            else:
                ext = os.path.basename(full_filename)
                ext = os.path.splitext(ext)[-1]
                resizetxt = os.path.basename(full_filename)[0:6]

                if ext == '.png' and resizetxt == 'Resize' and emo != 8:
                    global count
                    img_bgr = cv2.imread(full_filename)
                    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
                    img_gray = imutils.rotate_bound(img_gray, angle)
                    detected_faces = face_detector(img_gray, 1)
                    faces = [(x.left(), x.top(),
                              x.right(), x.bottom()) for x in detected_faces]
                    count = 0
                    if faces:
                        for n, face_rect in enumerate(faces):
                            if count >= 1:
                                break
                            else:
                                face = Image.fromarray(img_gray).crop(face_rect)
                                count += 1
                        face = face.resize((128, 128))
                        img_gray = np.array(face).reshape(128, 128)
                        X.append(img_gray)
                        Y.append(get_emo_int(full_filename))
        return X, Y
    except PermissionError:
        pass


def getXYckplus_load_flip(dirname, X, Y, file_path):
    try:
        filenames = listdir(dirname)
        for filename in filenames:
            full_filename = os.path.join(dirname, filename)
            emo = get_emo_int(full_filename)
            if os.path.isdir(full_filename):
                getXYckplus_load_flip(full_filename, X, Y, file_path)
            else:
                ext = os.path.basename(full_filename)
                ext = os.path.splitext(ext)[-1]
                resizetxt = os.path.basename(full_filename)[0:6]

                if ext == '.png' and resizetxt == 'Resize' and emo != 8:
                    img_bgr = cv2.imread(full_filename)
                    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
                    img_gray = cv2.flip(img_gray, 1)
                    img_gray = img_gray.reshape((128, 128))
                    X.append(img_gray)
                    Y.append(get_emo_int(full_filename))
                    file_path.append(full_filename)
        return X, Y, file_path
    except PermissionError:
        pass


def getXYckplus_load_noise(dirname, X, Y):
    try:
        filenames = listdir(dirname)
        for filename in filenames:
            full_filename = os.path.join(dirname, filename)
            emo = get_emo_int(full_filename)
            if os.path.isdir(full_filename):
                getXYckplus_load_noise(full_filename, X, Y)
            else:
                ext = os.path.basename(full_filename)
                ext = os.path.splitext(ext)[-1]
                resizetxt = os.path.basename(full_filename)[0:6]

                if ext == '.png' and resizetxt == 'Resize' and emo != 8:
                    global count
                    img_bgr = cv2.imread(full_filename)
                    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
                    img_gray = sp_noise(img_gray, 0.01)
                    img_gray = img_gray.reshape((128, 128))
                    X.append(img_gray)
                    Y.append(get_emo_int(full_filename))
        return X, Y
    except PermissionError:
        pass


def getXYckplus_LBP(img_grays):
    X = []
    for f in range(len(img_grays)):
        img_grays[f] = cv2.bilateralFilter(img_grays[f], 9, 20, 20)
        img_lbp = np.zeros((128, 128), np.uint8)
        for i in range(0, 128):
            for j in range(0, 128):
                img_lbp[i, j] = lbp.lbp_calculated_pixel(img_grays[f], i, j)  # 각 픽셀마다 lbp 계산해서 배열에 입력.

        X.append(img_lbp)
    return X


'''def getXYckplus_LBP(dirname,X,Y):
    #NUM = np.zeros((309,2304))
    try:
        filenames = listdir(dirname)
        for filename in filenames:

            full_filename = os.path.join(dirname, filename)
            emo = get_emo_int(full_filename)
            if os.path.isdir(full_filename):
                #convert_ckplus_files(full_filename)
                getXYckplus_LBP(full_filename,X,Y)
                #print('full_filename:',full_filename)
            else:
                #ext = os.path.splitext(full_filename)[-1]
                ext = os.path.basename(full_filename)
                ext = os.path.splitext(ext)[-1]
                resizetxt = os.path.basename(full_filename)[0:6]

                if ext == '.png' and resizetxt == 'Resize' and emo != 8:

                    #print('emo: ',emo)
                    #im = Image.open(full_filename)
                    #print('full_filename:',full_filename)
                    pixel=[]
                    #pixelnp=np.arange(48*48)
                    global count
                    img_bgr = cv2.imread(full_filename)
                    height, width, channel = img_bgr.shape
                    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

                    img_lbp = np.zeros((height, width), np.uint8)
                    for i in range(0, height):
                        for j in range(0, width):
                             img_lbp[i, j] = lbp.lbp_calculated_pixel(img_gray, i, j) #각 픽셀마다 lbp 계산해서 배열에 입력.
                    #for x in range(0, 128):
                    #    for y in range(0, 128):
                            #print(type(im.getpixel((y,x))))
                    #        pixel.append(img_lbp[x, y])
                    #print('pixel: ',pixelnp.shape)
                    X.append(img_lbp)
                    #X[count].append(int(pixel))
                    #print('imagepixel: ',X)
                    Y.append(get_emo_int(full_filename))
                    #print(Y)
                    #print(full_filename)
        #print('XSHAPE: ',len(X))
        #print(Y)
        return X,Y
    except PermissionError:
        pass
'''


def getXYckplus(dirname, X, Y):
    # NUM = np.zeros((309,2304))
    try:
        filenames = listdir(dirname)
        for filename in filenames:

            full_filename = os.path.join(dirname, filename)
            emo = get_emo_int(full_filename)
            if os.path.isdir(full_filename):
                # convert_ckplus_files(full_filename)
                getXYckplus(full_filename, X, Y)
                # print('full_filename:',full_filename)
            else:
                # ext = os.path.splitext(full_filename)[-1]
                ext = os.path.basename(full_filename)
                ext = os.path.splitext(ext)[-1]
                resizetxt = os.path.basename(full_filename)[0:6]

                if ext == '.png' and resizetxt == 'Resize' and emo != 8:

                    # print('emo: ',emo)
                    im = Image.open(full_filename)
                    # print('full_filename:',full_filename)
                    pixel = []
                    # pixelnp=np.arange(48*48)
                    for x in range(0, 128):
                        for y in range(0, 128):
                            # print(type(im.getpixel((y,x))))
                            pixel.append(im.getpixel((y, x)))
                    # print('pixel: ',pixelnp.shape)
                    X.append(pixel)
                    # X[count].append(int(pixel))
                    # print('imagepixel: ',X)
                    Y.append(get_emo_int(full_filename))
                    # print(Y)
                    # print(full_filename)
        # print('XSHAPE: ',len(X))
        # print(Y)
        return X, Y
    except PermissionError:
        pass


def getDatackplus():
    # images are 48x48 = 2304 size vectors
    # N = 35887
    # SUB파일 찾아가면서 append

    dirname = 'ckplus/extended-cohn-kanade-images/cohn-kanade-images/'
    # delete_ckplus_files(dirname)
    # convert_ckplus_files(dirname)
    X = []
    Y = []
    # X=[]
    X, Y = getXYckplus(dirname, X, Y)
    # print(X[0])
    # print(Y)
    # print(type(X))
    X, Y = np.array(X) / 255.0, np.array(Y)
    # print('getDataCK+ X.shape',X.shape)
    '''if balance_ones:
        # balance the 1 class
        X0, Y0 = X[Y!=1, :], Y[Y!=1]
        X1 = X[Y==1, :]
        X1 = np.repeat(X1, 9, axis=0)
        X = np.vstack([X0, X1])
        Y = np.concatenate((Y0, [1]*len(X1)))'''
    # print('이미지갯수:',len(X),len(Y))
    # print('getDataCK+ X.shape',X.shape)
    return X, Y


def getDatackplus_LBP(change):
    # images are 48x48 = 2304 size vectors
    # N = 35887
    # SUB파일 찾아가면서 append

    dirname = 'ckplus/extended-cohn-kanade-images/cohn-kanade-images/'
    # delete_ckplus_files(dirname)
    # convert_ckplus_files(dirname)
    X = []
    Y = []
    file_path = []
    img_gray = []
    if change == 'gray':
        img_gray, Y, file_path = getXYckplus_load_gray(dirname, X, Y, file_path)
        X = getXYckplus_LBP(img_gray)

    elif change == 'rotation5':
        img_gray, Y = getXYckplus_load_rotation(dirname, X, Y, 5)
        X = getXYckplus_LBP(img_gray)

    elif change == 'rotation-5':
        img_gray, Y = getXYckplus_load_rotation(dirname, X, Y, -5)
        X = getXYckplus_LBP(img_gray)

    elif change == 'flip':
        img_gray, Y, file_path = getXYckplus_load_flip(dirname, X, Y, file_path)
        X = getXYckplus_LBP(img_gray)

    elif change == 'noise':
        img_gray, Y = getXYckplus_load_noise(dirname, X, Y)
        X = getXYckplus_LBP(img_gray)

    X, Y, file_path = np.array(X) / 255.0, np.array(Y), np.array(file_path)
    return X, Y, file_path


def sort_by_subject(X, Y, file_path):
    X, Y, file_path = shuffle(X, Y, file_path)
    file_name = []
    sort_list = []
    X_new = []
    Y_new = []
    file_list_new = []
    for i in range(len(file_path)):
        name = int(os.path.basename(file_path[i])[8:11])
        print(name)
        file_name += [name]
    file_name = np.array(file_name)

    N, D1, D2 = X.shape
    X = X.reshape(N, D1 * D2)
    for idx in range(len(file_name)):
        sort_list.append((file_name[idx], X[idx], Y[idx], file_path[idx]))
    print(sort_list[0])
    sort_list = sorted(sort_list, key=lambda x: x[0])

    # sort_list = sorted(sort_list, key=lambda a_entry: a_entry[0])
    # for name,x,y,path in sort_list:
    #    X = np.vstack((X, x))
    for idx2 in range(len(file_name)):
        X_new.append(sort_list[idx2][1])
        Y_new.append(sort_list[idx2][2])
        file_list_new.append(sort_list[idx2][3])
    X_new = np.array(X_new)
    print('x new shape', X_new.shape)
    N, D = X_new.shape
    X_new = X_new.reshape(N, int(np.sqrt(D)), int(np.sqrt(D)))
    Y_new = np.array(Y_new).reshape(-1, 1)
    file_list_new = np.array(file_list_new)
    print('some data', file_list_new)
    print('shape', X_new.shape, Y_new.shape, file_list_new.shape)
    return X_new, Y_new, file_list_new


def getImageDatackplus():  # cnn main에서 불러오는 부분...
    X, Y = getDatackplus()
    N, D = X.shape
    # D, N = X.shape
    # print('getImageDataCK+ X.shape',X.shape)
    d = int(np.sqrt(D))
    X = X.reshape(N, 1, d, d)
    return X, Y


def getImageDatackplus_LBP(change):  # cnn main에서 불러오는 부분...
    X, Y, file_path = getDatackplus_LBP(change)
    X, Y, file_path = data_arg(X, Y, file_path)
    #X, Y, file_path = sort_by_subject(X, Y, file_path)
    # N, D1, D2  = X.shape
    # D, N = X.shape
    # print('getImageDataCK+ X.shape',X.shape)
    # d = int(np.sqrt(D))
    # X, Y, file_path = shuffle(X, Y, file_path)
    X = X.reshape(len(Y), 1, 128, 128)
    return X, Y, file_path


def data_arg(X, Y, file_path):
    new_X = []
    new_Y = []
    new_file_path = []
    for i in range(len(Y)):
        '''if int(Y[i])==0:
            new_X.append(X[i])
            new_Y.append(Y[i])
            new_file_path.append(file_path[i])'''
        if int(Y[i]) == 2 or int(Y[i]) == 4:
            new_X.append(X[i])
            new_Y.append(Y[i])
            new_file_path.append(file_path[i])

    X = np.concatenate((X, new_X))
    Y = np.concatenate((Y, new_Y))
    file_path = np.concatenate((file_path, new_file_path))
    X, Y, file_path = shuffle(X, Y, file_path)
    return X, Y, file_path


if __name__ == '__main__':
    Y = []
    X = []
    # convert_ckplus_files(id, dir)
    dirname = 'ckplus/extended-cohn-kanade-images/cohn-kanade-images/'

    getDatackplus()
    # convert_ckplus_files(dir)