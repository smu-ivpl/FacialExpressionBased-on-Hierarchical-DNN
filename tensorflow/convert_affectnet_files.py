import csv
import os
import pandas as pd
from os import listdir
from os.path import isfile, join
import re
from PIL import Image
import numpy as np
from glob import glob
import lbp as lbp
import cv2
from crop_images import faceImageCrop
import lbp as lbp
import cv2

EMOS = {"6": 0, "5": 1, "4": 2, "1": 3, "2": 4, "3": 5,"0": 6}

def get_emo_int(emo_value):
    emo_key = str(emo_value)
    return EMOS[emo_key]

def loading_batch_file(face_csv_data,batch_size,batch_n,emo_divide_point):
    file_list = []
    emo_list = []
    batch_start = 0+(batch_n*batch_size)
    batch_end = batch_start+batch_size
    print('Emotion_divide_point:',emo_divide_point)
    for i in range(0,7):
        #print('i번째:',i)
        start = batch_start + emo_divide_point[i]
        #print('start:',start)
        end = batch_end + emo_divide_point[i]
        #print('end:',end)
        #face_csv_data['subDirectory_filePath'] = face_csv_data['subDirectory_filePath'].map(lambda x: (face_data_img_path+x))
        if i==0:
            file_list.append(face_csv_data.loc[start:end-1,'subDirectory_filePath'])
            emo_list.append(face_csv_data.loc[start:end-1,'expression'])
            file_list = np.array(file_list).reshape(batch_size,)
            emo_list = np.array(emo_list).reshape(batch_size,)
        else:
            file_list=np.concatenate((file_list, face_csv_data.loc[start:end-1,'subDirectory_filePath']))
            emo_list=np.concatenate((emo_list,face_csv_data.loc[start:end-1,'expression']))

    #print('파일 SHAPE: ',file_list.shape)
    return file_list, emo_list

def loading_Manually_Annotated_file_lists_training(isLBP_str):
    face_data_csv_path='AffectNet/Manually_Annotated_file_lists/training.csv'
    face_data_img_path='AffectNet/Manually_Annotated_compressed/Manually_Annotated/Manually_Annotated_Images/'
    print('CSV파일 로딩...')
    face_csv_data=pd.read_csv(face_data_csv_path, sep=',')
    print('CSV파일 SORTING...\n')
    face_csv_data=face_csv_data.sort_values(by=["expression"])
    face_csv_data = face_csv_data.loc[face_csv_data['expression']<7,:]
    face_csv_data = face_csv_data.loc[face_csv_data['valence'] != -2,:]
    face_csv_data = face_csv_data.loc[face_csv_data['face_width'] != 0, :]
    face_csv_data = face_csv_data.loc[face_csv_data['face_height'] != 0, :]
    face_csv_data = face_csv_data.reset_index(drop=True)
    #print(face_csv_data.head(10))
    emo_divide_point=[0,0,0,0,0,0,0]
    count=0
    print('AffectNet Data Size:',face_csv_data.shape[0],'\n')
    for i in range(face_csv_data.shape[0]):
        #print('값확인:',int(face_csv_data.ix[i,"expression"]))
        if i==0 and (count is not 7):
            emo_divide_point[0]=0
            count+=1

        elif (int(face_csv_data.loc[i-1,"expression"]) != int(face_csv_data.loc[i,"expression"])) and (count is not 7):
            emo_divide_point[count]=i
            count+=1

    print('BATCH 파일 로딩...')
    batch_size=150
    print('BATCH 파일 SIZE: ', batch_size)
    batch_n=1
    file_list, emo_list = loading_batch_file(face_csv_data,batch_size,batch_n,emo_divide_point)
    for i, file in enumerate(file_list):
        file_list[i] = (face_data_img_path+file)
    Y = []
    X = []

    '''for i in range(len(face_csv_data['subDirectory_filePath'])):
        full_file_path=face_data_img_path+face_csv_data['subDirectory_filePath'][i]
        file_list.append(full_file_path)
        emo_list.append(face_data_csv['expression'])'''

    print('TRAINING DATA 길이:',len(file_list),'\n')
    new_file_list=[]
    if(isLBP_str=='LBP'):
        print('LBP Image loading start...')
        for idx, files in enumerate(file_list):
            pixel = []
            im = Image.open(files)
            im = faceImageCrop(files)
            if im is not -1:
                #im = im.convert('L')
                im = im.resize((128, 128))
                #print(im)
                im = np.array(im).reshape(128,128,3)
                height, width = 128,128
                #height, width, channel = img_bgr.shape
                img_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
                #img_gray = cv2.medianBlur(img_gray, 5)
                img_gray = cv2.bilateralFilter(img_gray, 9, 20, 20)
                img_lbp = np.zeros((height, width), np.uint8)
                for i in range(0, height):
                    for j in range(0, width):
                        img_lbp[i, j] = lbp.lbp_calculated_pixel(img_gray, i, j)  # 각 픽셀마다 lbp 계산해서 배열에 입력.

                X.append(img_lbp)
                Y.append(get_emo_int(emo_list[idx]))
                new_file_list.append(file_list[i])
            else:
                continue

    else:
        for idx, files in enumerate (file_list):
            #print('File name:',files)
            im = Image.open(files)
            im = faceImageCrop(files)
            im = im.convert('L')
            im = im.resize((128, 128))
            im_gray = np.array(im).reshape(128,128)
            X.append(im_gray)
            Y.append(get_emo_int(emo_list[idx]))

    X, Y, new_file_list = np.array(X) / 255.0, np.array(Y), np.array(new_file_list)
    N, D1, D2  = X.shape
    # D, N = X.shape
    # print('getImageDataCK+ X.shape',X.shape)
    #d = int(np.sqrt(D))
    X = X.reshape(N, 1, D1, D2 )
    print('LBP IMAGE DATA LOADING 완료\n')
    return X, Y, new_file_list


def loading_Manually_Annotated_file_lists_validation():
    face_data_csv_path = 'AffectNet/Manually_Annotated_file_lists/validation.csv'
    face_data_img_path = 'AffectNet/Manually_Annotated_compressed/Manually_Annotated/Manually_Annotated_Images/'
    print('CSV파일 로딩...')
    face_csv_data = pd.read_csv(face_data_csv_path, sep=',')
    print('CSV파일 SORTING...')
    face_csv_data = face_csv_data.sort_values(by=["expression"])
    face_csv_data = face_csv_data.loc[face_csv_data['expression'] < 7, :]
    face_csv_data = face_csv_data.reset_index(drop=True)
    # print(face_csv_data.head(10))
    emo_divide_point = [0, 0, 0, 0, 0, 0, 0]
    count = 0
    print('face_csv_data size:', face_csv_data.shape[0])
    for i in range(face_csv_data.shape[0]):
        # print('값확인:',int(face_csv_data.ix[i,"expression"]))
        if i == 0 and (count is not 7):
            emo_divide_point[0] = 0
            count += 1

        elif (int(face_csv_data.loc[i - 1, "expression"]) != int(face_csv_data.loc[i, "expression"])) and (
                count is not 7):
            emo_divide_point[count] = i
            count += 1

    print('BATCH 파일 로딩...')
    batch_size = 30
    batch_n = 0
    file_list, emo_list = loading_batch_file(face_csv_data, batch_size, batch_n, emo_divide_point)
    Y = []
    X = []

    '''for i in range(len(face_csv_data['subDirectory_filePath'])):
        full_file_path=face_data_img_path+face_csv_data['subDirectory_filePath'][i]
        file_list.append(full_file_path)
        emo_list.append(face_data_csv['expression'])'''

    print('TRAINING DATA 길이:', len(file_list))

    for idx, files in enumerate(file_list):
        pixel = []
        # print('File name:',files)
        im = Image.open(face_data_img_path + files)
        im = faceImageCrop(face_data_img_path + files)
        im = im.convert('L')
        im = im.resize((128, 128))
        for x in range(0, 128):
            for y in range(0, 128):
                pixel.append(im.getpixel((y, x)))
        X.append(pixel)
        Y.append(get_emo_int(emo_list[idx]))

    X, Y = np.array(X) / 255.0, np.array(Y)
    print('IMAGE DATA LOADING 완료')
    return X, Y

def loading_Automatically_Annotated_file_lists():
    face_data_csv_path = 'AffectNet/Automatically_Annotated_file_list/automatically_annotated.csv'
    face_data_img_path = 'AffectNet/Automatically_Annotated_compressed/Automatically_Annotated/Automatically_Annotated_Images/'
    face_csv_data = pd.read_csv(face_data_csv_path, sep=',')
    Y = []
    X = []

    file_list = []
    emo_list = []
    for i in range(len(face_csv_data['subDirectory_filePath'])):
        full_file_path = face_data_img_path + face_csv_data['subDirectory_filePath'][i]
        file_list.append(full_file_path)
        emo_list.append(face_data_csv['expression'])

    print('automatically data lenght:', len(file_list))

    for idx, files in enumerate(file_list):
        pixel = []
        im = Image.open(files)
        im = im.resize((128, 128))
        for x in range(0, 128):
            for y in range(0, 128):
                pixel.append(im.getpixel((y, x)))
        X.append(pixel)
        Y.append(get_emo_int(emo_list[i]))

    X, Y = np.array(X) / 255.0, np.array(Y)
    return X, Y

if __name__ == '__main__':
    loading_Manually_Annotated_file_lists_training()