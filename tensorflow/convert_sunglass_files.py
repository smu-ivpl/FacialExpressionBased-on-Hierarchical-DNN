import os
from os import listdir
from os.path import isfile, join
import re
from PIL import Image
import numpy as np
import lbp as lbp
import cv2
from crop_images import faceImageCrop
file_dir = 'FacesWithGlasses/'


# convert tiff to png
def convert_sunglass_files():
    onlyfiles = [join(file_dir, f) for f in listdir(file_dir) if isfile(join(file_dir, f))]
    png_files = [f for f in onlyfiles if f.endswith(".jpg")]

    for file_name in png_files:
        if file_name.find('Resize_') is -1:
            # full_filename = os.path.join(file_dir, file_name)
            # print('dir: '+full_filename)
            im = Image.open(file_name)
            im = im.resize((128, 128))
            title, ext = os.path.splitext(os.path.basename(file_name))
            newname = 'Resize_' + title
            im.convert("L").save(os.path.join(file_dir, newname + ext))

def getDataSunglass():
    # images are 48x48 = 2304 size vectors
    convert_sunglass_files()
    X = []

    file_list = listdir(file_dir)
    for files in file_list:
        pixel = []
        if files.find('Resize_') is not -1:
            im = Image.open(file_dir + files)
            for x in range(0, 128):
                for y in range(0, 128):
                    pixel.append(im.getpixel((y, x)))
            X.append(pixel)
            # X[count].append(int(pixel))
            # print('imagepixel: ',X)
            # print('label:',get_emo_int_binary(files))
    X = np.array(X) / 255.0
    # print('getDataJaffe X.shape',X.shape)

    # print('이미지갯수:',len(X),len(Y))
    return X
def getDataSunglass_32():
    # images are 48x48 = 2304 size vectors
    convert_sunglass_files()
    X = []

    file_list = listdir(file_dir)
    for files in file_list:
        pixel = []
        if files.find('Resize_') is not -1:
            im = Image.open(file_dir + files)
            im = faceImageCrop(file_dir + files)
            im = im.resize((32,32))
            for x in range(0, 32):
                for y in range(0, 32):
                    pixel.append(im.getpixel((y, x)))
            X.append(pixel)
            # X[count].append(int(pixel))
            # print('imagepixel: ',X)
            # print('label:',get_emo_int_binary(files))
    X = np.array(X) / 255.0
    # print('getDataJaffe X.shape',X.shape)

    # print('이미지갯수:',len(X),len(Y))
    return X

def getImageDataSunglass():  # cnn main에서 불러오는 부분...
    X = getDataSunglass_32()
    N, D = X.shape
    # D, N = X.shape
    # print('getImageDataJaffe X.shape',X.shape)
    d = int(np.sqrt(D))
    X = X.reshape(N, 1, d, d)
    return X

if __name__ == '__main__':
    # convert_jaffe_files(id, dir)
    getDataSunglass()