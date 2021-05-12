import numpy as np
from keras.utils import np_utils
from Autoencoder_keras import Autoencoder
from keras.models import load_model
import itertools
import get_dataset as get_db
from itertools import combinations
import get_dataset as getdb
import convert_ckplus_files as ck
import convert_jaffe_files as jaffe
import convert_ferg_files as ferg
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from keras.callbacks import ModelCheckpoint
from keras.models import Model
from keras import backend as K
from util import count_dataset
import os
from crop_images import faceImageCrop
import cv2
import scipy.misc as sc
from PIL import Image
dirname_ck = 'ckplus/extended-cohn-kanade-images/cohn-kanade-images/'
import keras.backend as K
from os import environ
from importlib import reload
def set_keras_backend(backend):
    if K.backend() != backend:
        environ['KERAS_BACKEND'] = backend
        reload(K)
        assert K.backend() == backend
set_keras_backend('theano')
K.set_image_dim_ordering('th')
def loading_data():
    X_j, Y_j = jaffe.getImageDataJaffe()
    count2 = count_dataset(Y_j)
    print('>JAFFE Dataset')
    print('Angry:    ', count2[0])
    print('Disgust:  ', count2[1])
    print('Fear:     ', count2[2])
    print('Happy:    ', count2[3])
    print('Sad:      ', count2[4])
    print('Surprise: ', count2[5])
    print('Neutral:  ', count2[6])
    print('total:    ', sum(count2))
    emo_list_path=[]
    emo_list = []
    neutral_list_path=[]
    neutral_list = []
    X=[]
    Y=[]
    X, Y = get_db.get_jaffe_path(X, Y)
    for i in range(len(X)):
        if int(Y[i]) != 6:
            emo_list.append(X_j[i])
            emo_list_path.append(X[i])
    for e in range(len(emo_list_path)):
        count=0
        for j in range(len(X)):
            if os.path.basename(emo_list_path[e])[:2] == os.path.basename(X[j])[:2] and int(Y[j]) == 6 and count==0:
                neutral_list_path.append(X[j])
                neutral_list.append(X_j[j])
                count+=1
    print('emo_list 길이:',len(emo_list),'neutral_list 길이:',len(neutral_list))
    print('EX) Emotion path:', emo_list_path[3], 'neutral_list 길이:', neutral_list_path[3])
    print('--------------------JAFFE LOADING 완료------------------------')
    X_c, Y_c = ck.getImageDatackplus()
    count3 = count_dataset(Y_c)
    print('>CK+ Dataset')
    print('Angry:    ', count3[0])
    print('Disgust:  ', count3[1])
    print('Fear:     ', count3[2])
    print('Happy:    ', count3[3])
    print('Sad:      ', count3[4])
    print('Surprise: ', count3[5])
    print('Neutral:  ', count3[6])
    print('total:    ', sum(count3))

    emo_list_path_c=[]
    neutral_list_c=[]
    X = []
    Y = []
    X, Y = get_db.get_ck_plus_path(dirname_ck,X,Y)
    for i in range(len(X)):
        if int(Y[i]) != 6:
            emo_list.append(X_c[i])
            emo_list_path_c.append(X[i])
    for e in range(len(emo_list_path_c)):
        count=0
        for j in range(len(X)):
            if os.path.basename(emo_list_path_c[e])[:15] == os.path.basename(X[j])[:15] and int(Y[j]) == 6 and count==0:
                neutral_list_c.append(X[j])
                neutral_list.append(X_c[j])
                count+=1
    print('emo_list 길이:',len(emo_list),'neutral_list 길이:',len(neutral_list))
    print('EX) Emotion path:', emo_list_path_c[3], 'neutral_list 길이:', neutral_list_c[3])
    print('--------------------CK+ LOADING 완료------------------------')

    return np.array(emo_list), np.array(neutral_list)

def train():
    
    
    X = []
    X_n = []
    X = os.listdir('afew_test/angry/')
    X_n = os.listdir('afew_test/neutral/')
#    X, X_n = loading_data()
    
    #train_N = int((len(Y) / 100) * 90)

    print('X.shape',X.shape)
    print('X_n.shape',X_n.shape)
    #X = X.transpose((0, 2, 3, 1))
    #X=X.reshape((-1, 128, 128, 1))
    #X_n = X_n.transpose((0, 2, 3, 1))
    #X_n = X_n.reshape((-1, 128, 128, 1))
    # output labels should be one-hot vectors - ie,
    # 0 -> [0, 0, 1]
    # 1 -> [0, 1, 0]
    # 2 -> [1, 0, 0]
    # define optimizer and objective, compile cnn
#    model = Autoencoder('Autoencoder/model/0612/weights_best_99_0.04.hdf5')
    model = Autoencoder()
    model.compile(loss="mean_squared_error", optimizer="adam", metrics=['accuracy'])
    model.summary()
    # checkpoint
    # filepath = "simple_cnn/model/weights_best_{epoch:02d}_{val_acc:.2f}.hdf5"
    filepath = "Autoencoder/model/weights_test_{epoch:02d}_{val_acc:.2f}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=False, mode='auto', period=5)
    callbacks_list = [checkpoint]
    # train
    history = model.fit(X, X_n, validation_split=0.3, epochs=1000, callbacks=callbacks_list, verbose=1)

    # summarize history for accuracy
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model accuracy and loss')
    plt.ylabel('accuracy/loss')
    plt.xlabel('epoch')
    plt.legend(['train_acc', 'test_acc', 'train_loss', 'test_loss'], loc='upper left')
    plt.savefig("Autoencoder/graph/train_result_graph.png")

    print('==finish training==')

    '''score = cnn.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])'''
    # cnn.save("cnn_model_9_1.h5")

def get_neutral_image(path,model):
    set_keras_backend('theano')
    K.set_image_dim_ordering('th')
    im = faceImageCrop(path)
    if im is not -1:
        im = im.convert('L')
        im = im.resize((128, 128))

        pixel = []
        X = []
        for x in range(0, 128):
            for y in range(0, 128):
                pixel.append(im.getpixel((y, x)))
        X.append(pixel)
        X = np.array(X) / 255.0

        N, D = X.shape
        d = int(np.sqrt(D))
        X = X.reshape(N, 1, d, d)


        out = model.predict(X)
        #print('Result shape:', out.shape)

        N, c, d, d = out.shape
        D = int(d * d)
        out = out.reshape(N, D)
        out = out.reshape(128, 128)
        #im = sc.toimage(out, mode='L')
        #im = np.array(out, dtype=np.float32)
        #im_bgr=cv2.cvtColor(im,cv2.COLOR_GRAY2BGR)
        im = np.array(out*255, dtype=np.uint8)
        #gray = cv2.cvtColor(im_bgr, cv2.COLOR_BGR2GRAY)

        #image = cv2.imdecode(np.fromstring(out, dtype=np.uint8), 1)
        print('Generate Neutral Image Complete')
        return im
        #print('gray image:',im)
    else:
        print('Fail generate neutral image')
        return -1

def test():
    model = Autoencoder('Autoencoder/model/weights_best_169_0.03.hdf5')
    path=[]
    #path = 'AffectNet/Manually_Annotated_compressed/Manually_Annotated/Manually_Annotated_Images/8/347adda0a2a1cec16b934d447d6d9e91eeb0651d38f086490f239f66.jpg'
    path.append('afew_test/angry.png')
    #path.append('affectnet_test/2.jpg')
    #path.append('affectnet_test/3.png')
    #path.append('affectnet_test/4.png')
    #path.append('affectnet_test/5.jpg')

    for i in range(len(path)):
        im = Image.open(path[i])

        #im = faceImageCrop(path[i])
        im = im.convert('L')
        im = im.resize((128, 128))

        pixel = []
        X = []
        '''for x in range(0, 128):
            for y in range(0, 128):
                pixel.append(im.getpixel((y, x)))'''
        pixel=(np.array(im)).reshape(128,128)
        X.append(pixel)
        X = np.array(X) / 255.0

        #N, D = X.shape
        #d = int(np.sqrt(D))
        X = X.reshape(-1, 1, 128, 128)


        out = model.predict(X)
        print('Result shape:', out.shape)

        N, c, d, d = out.shape
        D = int(d * d)
        out = out.reshape(N, D)

        plt.imshow(out.reshape(128, 128), cmap='gray')
        plt.title('result')
        plt.show()

if __name__ == '__main__':
    test()