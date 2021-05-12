# -*- coding: utf-8 -*-
"""
Created on Wed Feb  7 14:04:04 2018

@author: KIMJIHAE
"""

import cv2
import numpy as np
import dlib
from matplotlib import pyplot as plt
from randmark_detect import shape_to_np, get_img
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

def show_output(output_list, g_vec, diff_g_vec, neutral_g_vec):  # lbp 결과 그래프와 이미지로 보여줌.
    output_list_len = len(output_list)
    figure = plt.figure()
    for i in range(output_list_len):
        current_dict = output_list[i]
        current_img = current_dict["img"]
        current_xlabel = current_dict["xlabel"]
        current_ylabel = current_dict["ylabel"]
        current_xtick = current_dict["xtick"]
        current_ytick = current_dict["ytick"]
        current_title = current_dict["title"]
        current_type = current_dict["type"]
        current_plot = figure.add_subplot(1, output_list_len, i + 1)
        if current_type == "gray":
            current_plot.imshow(current_img, cmap=plt.get_cmap('gray'))
            current_plot.set_title(current_title)
            current_plot.set_xticks(current_xtick)
            current_plot.set_yticks(current_ytick)
            current_plot.set_xlabel(current_xlabel)
            current_plot.set_ylabel(current_ylabel)

            lst_x, lst_y = [], []
            for i in range(0, len(neutral_g_vec)):
                if (i % 2) == 0:
                    lst_x.append(neutral_g_vec[i])
                else:
                    lst_y.append(neutral_g_vec[i])

            current_plot.plot(lst_x, lst_y, 'ro', markersize=2, color='white')

            for i in range(0, len(g_vec)):
                if (i % 2) == 0:
                    if diff_g_vec[i] < 0 or diff_g_vec[i + 1] < 0:
                        current_plot.arrow(g_vec[i], g_vec[i + 1], diff_g_vec[i], diff_g_vec[i + 1], fc="r", ec="r",
                                           head_width=0.05, head_length=0.1)
                    else:
                        current_plot.arrow(g_vec[i], g_vec[i + 1], diff_g_vec[i], diff_g_vec[i + 1], fc="b", ec="b",
                                           head_width=0.05, head_length=0.1)

    plt.show()


def neutral_roi():
    image_file = 'FERG_DB_256/mery/mery_neutral/mery_neutral_1.png'  # 이미지 가지고 오는 부분
    ''''''


def img_load(filepath):
    # image_file = 'jaffe/AN_resize037210.png'
    image_file = filepath  # 이미지 가지고 오는 부분
    # image_file = 'ckplus/extended-cohn-kanade-images/cohn-kanade-images/S022/005/Resize_S022_005_00000032.png'
    img_bgr = cv2.imread(image_file)
    height, width, channel = img_bgr.shape
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    img_gray = cv2.equalizeHist(img_gray)  # 히스토그램 평활화

    return image_file, height, width, img_gray


def make_image_show(img_gray, g_vec, diff_g_vec, neutral_g_vec):
    output_list = []
    output_list.append({
        "img": img_gray,
        "xlabel": "",
        "ylabel": "",
        "xtick": [],
        "ytick": [],
        "title": "Gray Image",
        "type": "gray"
    })

    show_output(output_list, g_vec, diff_g_vec, neutral_g_vec)


def get_g_vec(shape):
    left_eyebrow = np.zeros(6, dtype=int)
    left_eyebrow = (shape[17][0], shape[17][1], shape[19][0], shape[19][1], shape[21][0], shape[21][1])

    right_eyebrow = np.zeros(6, dtype=int)
    right_eyebrow = (shape[22][0], shape[22][1], shape[24][0], shape[24][1], shape[26][0], shape[26][1])

    left_eye = np.zeros(8, dtype=int)
    left_eye = (
    shape[36][0], shape[36][1], int((shape[37][0] + shape[38][0]) / 2), int((shape[37][1] + shape[38][1]) / 2),
    shape[39][0], shape[39][1], int((shape[40][0] + shape[41][0]) / 2), int((shape[40][1] + shape[41][1]) / 2))

    right_eye = np.zeros(8, dtype=int)
    right_eye = (
    shape[42][0], shape[42][1], int((shape[43][0] + shape[44][0]) / 2), int((shape[43][1] + shape[44][1]) / 2),
    shape[45][0], shape[45][1], int((shape[46][0] + shape[47][0]) / 2), int((shape[46][1] + shape[47][1]) / 2))

    mouse = np.zeros(8, dtype=int)
    mouse = (
    shape[48][0], shape[48][1], shape[51][0], shape[51][1], shape[54][0], shape[54][1], shape[57][0], shape[57][1])

    g_vec = np.concatenate((left_eyebrow, right_eyebrow, left_eye, right_eye, mouse))

    return g_vec


def extract_geometry(filepath, filepath_neutral, neutral_image_path=True):
    # image_file, height, width, img_gray = img_load(filepath)
    detector = dlib.get_frontal_face_detector()
    image = cv2.imread(filepath)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # detect faces in the grayscale image
    rects = detector(gray, 1)
    #print(rects.shape)

    # image, gray, rects = get_img(filepath)
    if neutral_image_path == True:
        print('Using Existing Neutral Image')
        # image_file_neu, height_neu, width_neu, img_gray_neu = img_load(filepath_neutral)
        # rects=[]
        # rects_neu=[]
        # detector = dlib.get_frontal_face_detector()
        image_neu = cv2.imread(filepath_neutral)
        gray_neu = cv2.cvtColor(image_neu, cv2.COLOR_BGR2GRAY)

        # detect faces in the grayscale image
        print("===", gray_neu.shape)
        rects_neu = detector(gray_neu, 1)
        # image_neu, gray_neu, rects_neu = get_img(filepath_neutral)

    elif neutral_image_path == False:
        print('Using Generated Neutral Image')
        img_gray_neu = filepath_neutral
        # height_neu, width_neu, channel_neu = filepath_neutral.shape
        # detector = dlib.get_frontal_face_detector()
        if filepath_neutral is not -1:
            rects_neu = detector(img_gray_neu, 1)
            # print('rects_neu',rects_neu)
            gray_neu = img_gray_neu
            # rects_neu=rects[0]
        else:
            rects_neu = -1
    #print("here....")
    predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
    if (len(rects) is not 0) and (len(rects_neu) is not 0) and (rects_neu is not -1):
        rect = rects[0]
        rects_neu = rects_neu[0]
        shape = predictor(gray, rect)
        shape = shape_to_np(shape)

        shape_neutral = predictor(gray_neu, rects_neu) #predictor : landmark
        shape_neutral = shape_to_np(shape_neutral)
        print(shape_neutral.shape)

        g_vec = get_g_vec(shape)
        neutral_g_vec = get_g_vec(shape_neutral)

        diff_g_vec = np.subtract(neutral_g_vec, g_vec)
        print("diff_g_vec : ", diff_g_vec.shape)

        # make_image_show(img_gray,img_lbp,left_eye_lbp_block,right_eye_lbp_block,nose_lbp_block,mouse_lbp_block)
        # print('geometric feature: ',diff_g_vec)

        # print('geometric feature shape: ',diff_g_vec.shape)

        # make_image_show(img_gray, g_vec, diff_g_vec, neutral_g_vec)

        # print("Geometric feature extraction Program is finished")


        return diff_g_vec
    else:
        print('Cannot Capture Face')
        return -1


if __name__ == '__main__':
    # filepath = 'ckplus/extended-cohn-kanade-images/cohn-kanade-images/S005/001/Resize_S005_001_00000011.png'
    # filepath_neutral = 'ckplus/extended-cohn-kanade-images/cohn-kanade-images/S005/001/Resize_S005_001_00000001.png'
    filepath = 'FERG_DB_256/aia/aia_anger/aia_anger_10.png'
    filepath_neutral = 'FERG_DB_256/aia/aia_neutral/aia_neutral_10.png'
    extract_geometry(filepath, filepath_neutral)