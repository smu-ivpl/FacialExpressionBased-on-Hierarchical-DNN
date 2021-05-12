# -*- coding: utf-8 -*-
"""
Created on Mon Sep 11 14:19:53 2017

@author: kimjihae
"""
# jaffe이미지(tiff 256*256)를 48*48로 변형하고 다시 저장하는 과정.
from os import listdir
from os.path import isfile, join
import re
from PIL import Image
import numpy as np
from glob import glob
import lbp as lbp
import cv2
from crop_images import faceImageCrop
from util import sp_noise
import dlib
import imutils

id = 37210
dir = 'jaffe/'
EMOS = {"AN": 0, "DI": 1, "FE": 2, "HA": 3, "SA": 4, "SU": 5, "NE": 6}


# convert tiff to png
def convert_jaffe_files(id, dir):
    onlyfiles = [join(dir, f) for f in listdir(dir) if isfile(join(dir, f))]
    tiff_files = [f for f in onlyfiles if f.endswith(".tiff")]

    # KA.AN1.39.png
    # drop first two letters
    # AN    anger
    # DI    disgust
    # FE    fear
    # HA    happy
    # NE    neutral
    # SA    unhappy (previously sad)
    # SU    surprise

    for name in tiff_files:
        im = Image.open(name)
        # im = im.resize((128, 128))
        png_name = re.sub("\.tiff$", ".png", name)  # change suffix
        png_name = re.sub("\w\w\.AN\d\.\d\d?\d?", name[6:8] + "_" + "AN_change%06d" % id, png_name)
        png_name = re.sub("\w\w\.DI\d\.\d\d?\d?", name[6:8] + "_" + "DI_change%06d" % id, png_name)
        png_name = re.sub("\w\w\.FE\d\.\d\d?\d?", name[6:8] + "_" + "FE_change%06d" % id, png_name)
        png_name = re.sub("\w\w\.HA\d\.\d\d?\d?", name[6:8] + "_" + "HA_change%06d" % id, png_name)
        png_name = re.sub("\w\w\.NE\d\.\d\d?\d?", name[6:8] + "_" + "NE_change%06d" % id, png_name)
        png_name = re.sub("\w\w\.SA\d\.\d\d?\d?", name[6:8] + "_" + "SA_change%06d" % id, png_name)
        png_name = re.sub("\w\w\.SU\d\.\d\d?\d?", name[6:8] + "_" + "SU_change%06d" % id, png_name)

        # print ('converting %s to %s' % (name, png_name))
        id += 1
        im.save(png_name)


def get_emo_int(filename):
    # pieces = filename.split('/')
    emo_key = filename[3:5]
    return EMOS[emo_key]


def getDataJaffe():
    # images are 48x48 = 2304 size vectors
    # N = 35887
    id = 37210
    dir = 'jaffe/'
    # convert_jaffe_files(id, dir)
    Y = []
    X = []
    # first = True

    file_list = listdir(dir)
    for files in file_list:
        pixel = []
        # if first:
        #    first = False
        # else:
        if files.find('_change') is not -1:
            im = Image.open(dir + files)
            im = faceImageCrop(dir + files)
            # im = im.convert('L')
            im = im.resize((128, 128))
            # print(im)
            im = np.array(im).reshape(128, 128)
            height, width = 128, 128
            # img_bgr = cv2.imread(dir+files)
            # height, width, channel = img_bgr.shape
            img_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            for x in range(0, 128):
                for y in range(0, 128):
                    pixel.append(im.getpixel((y, x)))
            X.append(pixel)
            # X[count].append(int(pixel))
            # print('imagepixel: ',X)
            Y.append(get_emo_int(files))

    X, Y = np.array(X) / 255.0, np.array(Y)
    return X, Y


def getDataJaffe_rotation(angle):
    dir = 'jaffe/'
    Y = []
    X = []
    file_path=[]
    file_list = listdir(dir)
    face_detector = dlib.get_frontal_face_detector()
    for files in file_list:
        if files.find('_change') is not -1:
            im = Image.open(dir + files)
            im = np.array(im)
            img_gray = imutils.rotate_bound(im, angle)

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
            Y.append(get_emo_int(files))
            file_path.append(dir+files)
    return X, Y, file_path

def getDataJaffe_flip():
    dir = 'jaffe/'
    Y = []
    X = []
    file_path=[]
    file_list = listdir(dir)
    face_detector = dlib.get_frontal_face_detector()
    for files in file_list:
        if files.find('_change') is not -1:
            # im = Image.open(dir+files)
            im = Image.open(dir + files)
            #im = faceImageCrop(dir + files)
            # im = im.convert('L')
            #im = im.resize((128, 128))
            im = im.transpose(Image.FLIP_LEFT_RIGHT) # 좌우반전
            # print(im)
            img_gray = np.array(im)
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
            Y.append(get_emo_int(files))
            file_path.append(dir+files)
    return X, Y, file_path

def getDataJaffe_noise():
    dir = 'jaffe/'
    Y = []
    X = []
    file_path=[]
    file_list = listdir(dir)
    face_detector = dlib.get_frontal_face_detector()
    for files in file_list:
        pixel = []
        if files.find('_change') is not -1:
            # im = Image.open(dir+files)
            im = Image.open(dir + files)
            img_gray = sp_noise(np.array(im),0.01)
            #img_gray = np.array(im).reshape(128, 128)
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
            #im = faceImageCrop(dir + files)
            # im = im.convert('L')
            #im = im.resize((128, 128))
            X.append(img_gray)
            Y.append(get_emo_int(files))
            file_path.append(dir+files)
    return X, Y, file_path

def getDataJaffe_gray():
    dir = 'jaffe/'
    Y = []
    X = []
    file_path=[]
    file_list = listdir(dir)
    face_detector = dlib.get_frontal_face_detector()
    for files in file_list:
        if files.find('_change') is not -1:
            #im = Image.open(dir+files)
            im = Image.open(dir+files)
            img_gray = np.array(im)
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
            Y.append(get_emo_int(files))
            file_path.append(dir+files)
    return X, Y, file_path


def adjust_gamma(image, gamma):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")

    return cv2.LUT(image, table)


kernel_3x3 = np.ones((3, 3), np.float32) / 9
kernel_sharpening = np.array([[-1, -1, -1],
                              [-1, 9, -1],
                              [-1, -1, -1]])


def convertJaffe_LBP(img_gray):
    X = []
    for f in range(len(img_gray)):
        img_lbp = np.zeros((128, 128), np.uint8)
        img_gray[f] = adjust_gamma(img_gray[f], 1.5)
        #img_gray[f] = cv2.equalizeHist(img_gray[f])

        #img_gray[f] = cv2.medianBlur(img_gray[f], 5)
        #img_gray[f] = cv2.GaussianBlur(img_gray[f], (3, 3), 0)
        # img_gray[f] = cv2.filter2D(img_gray[f], -1, kernel_sharpening)
        img_gray[f] = cv2.bilateralFilter(img_gray[f], 9,20,20)
        for i in range(0, 128):
            for j in range(0, 128):
                img_lbp[i, j] = lbp.lbp_calculated_pixel(img_gray[f], i, j)  # 각 픽셀마다 lbp 계산해서 배열에 입력.
        X.append(img_lbp)
        # X.append(img_gray[f])
    return X


def getDataJaffe_LBP(change):
    X = []
    Y = []
    file_path = []
    img_gray = []
    if change == 'gray':
        img_gray, Y, file_path = getDataJaffe_gray()
        X = convertJaffe_LBP(img_gray)
        # X = img_gray

    elif change == 'rotation5':
        img_gray, Y, file_path = getDataJaffe_rotation(5)
        X = convertJaffe_LBP(img_gray)
        # X = img_gray

    elif change == 'rotation-5':
        img_gray, Y, file_path = getDataJaffe_rotation(-5)
        X = convertJaffe_LBP(img_gray)
        # X = img_gray

    elif change == 'flip':
        img_gray, Y, file_path = getDataJaffe_flip()
        X = convertJaffe_LBP(img_gray)
        # X = img_gray

    elif change == 'noise':
        img_gray, Y, file_path = getDataJaffe_noise()
        X = convertJaffe_LBP(img_gray)
        # X = img_gray

    X, Y, file_path = np.array(X) / 255.0, np.array(Y), np.array(file_path)
    return X, Y, file_path


def getImageDataJaffe():  # cnn main에서 불러오는 부분...
    X, Y = getDataJaffe()
    N, D = X.shape
    # D, N = X.shape
    # print('getImageDataJaffe X.shape',X.shape)
    d = int(np.sqrt(D))
    X = X.reshape(N, 1, d, d)
    return X, Y


def getImageDataJaffe_LBP(change):  # cnn main에서 불러오는 부분...
    X, Y, file_path = getDataJaffe_LBP(change)
    print(X.shape)
    N, D1, D2 = X.shape
    # D, N = X.shape
    # print('getImageDataJaffe X.shape',X.shape)
    # d = int(np.sqrt(D))
    X = X.reshape(N, 1, D1, D2)
    return X, Y, file_path


if __name__ == '__main__':
    convert_jaffe_files(id, dir)
    # getDataJaffe_LBP()