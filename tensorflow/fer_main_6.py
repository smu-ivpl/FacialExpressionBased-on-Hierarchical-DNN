import numpy as np
from keras.utils import np_utils
from cnn_keras_6 import CNN
from keras.models import load_model
import itertools
from Autoencoder_keras import Autoencoder
from itertools import combinations
import convert_ckplus_files as ck
import convert_jaffe_files as jaffe
import convert_affectnet_files as affect
import Autoencoder_main as auto_main
import get_dataset as get_db
import convert_ferg_files as ferg
import geometric as geo
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from keras.callbacks import ModelCheckpoint
from keras.models import Model
from geo_cnn_keras import shallow_CNN
from operator import add
import lbp as lbp
from train_app_6 import loading_n_data, get_n_dataset, count_dataset, draw_confusion_matrix, get_6_dataset
from keras import backend as K
from pylab import *
import os
from os import environ
from importlib import reload
from crop_images import faceImageCrop
import glob
import time
def set_keras_backend(backend):
    if K.backend() != backend:
        environ['KERAS_BACKEND'] = backend
        reload(K)
        assert K.backend() == backend
set_keras_backend('tensorflow')
K.set_image_dim_ordering('tf')

def fer_main(str):
    if str == 'new' :
        print('Predict New Image')

    elif str == 'CK+' :
        label = True
        for iter in range(1, 11):
            result_emotion(str, iter, label)
        print('Predict CK+ Image')

    elif str == 'JAFFE' :
        label=True
        #for iter in range(1,11):
        result_emotion(str, 1, label)
        print('Predict JAFFE Image')

    elif str == 'FERG' :
        label = True
        result_emotion(str, iter,label)
        print('Predict FERG Image')

    elif str == 'AffectNet' :
        label = True
        #for iter in range(1, 11):
        result_emotion(str, 1, label)
        print('Predict AffectNet Image')

    elif str == 'ALL100' :
        label = True
        result_emotion_all(str)
        print('Predict All db Image')
    else:
        print('error')

def result_emotion(str, iter, label=True):
    if label==True and (str=='CK+'or str=='JAFFE'):
        X = []
        Y = []
        weights = [0.8]
        #weights = [0.4]
        ####npy 파일로 test image loading
        X, Y = image_load(str, iter)
        X=X.reshape(-1,128,128,1)
        Y=Y.reshape(-1,1)

        '''count_emotions1 = count_dataset(Y)
        f = open('result_0919/08_top2_model_use_testData_count_{0}_{1}.txt'.format((str),(iter)), 'w')
        f.write('{0} Dataset Test \n'.format((str)))
        f.write('Angry:    %d \n' % count_emotions1[0])
        f.write('Disgust:  %d \n' % count_emotions1[1])
        f.write('Fear:     %d \n' % count_emotions1[2])
        f.write('Happy:    %d \n' % count_emotions1[3])
        f.write('Sad:      %d \n' % count_emotions1[4])
        f.write('Surprise: %d \n' % count_emotions1[5])
        f.write('total:    %d \n' % sum(count_emotions1))
        f.close()'''

        print('Appearance Model Loading...')
        app_model = app_model_load(str, iter)
        print('Appearance Model Loading Complete')

        for w in range(len(weights)):
            count = 0
            Pred = np.zeros(len(Y))
            time_count = 0
            for i in range(0,50):
                #if i%5==0:
                ###predict time start
                start_t = time.time()
                appearance_result_list, top2_list = appearance_cnn(app_model, X[i], Y[i])
                print('Appearance Feature Extraction Complete')
                automodel=0
                geometric_result_list = geometric_cnn(str,i, iter, top2_list[0], top2_list[1],automodel)
                print('Geometric Feature Extraction Complete')
                ###predict time end
                end_t = time.time()
                time_count += (end_t - start_t)
                if geometric_result_list is not -1:
                    combine_result = joint_feature(appearance_result_list, top2_list, geometric_result_list,weights[w])
                    print('Combine Feature Extraction Complete')
                else:
                    combine_result = appearance_result_list
                    print('Only using Appearance Feature')

                print('Per image time:', float(time_count / len(Y)))
                topk = np.argsort(combine_result)
                top1 = int(topk[5])
                Pred[i]=top1
                print('Emotion {0}: '.format(i), top1)
                print('Emotion {0} real:'.format(i), Y[i])
                print('----------------------------------------')
                if int(Y[i]) == int(top1):
                    count += 1
            ### ---------------
            # CONFUSION MATRIX
            ### ---------------
            #draw_confusion_matrix(Y, Pred, str+'08top2model',iter)
            #print('Save Confusion Matrix')

            #weight test save
            #print('Correct FER score (%): ', (count / len(Y)) * 100)
            #f = open('result_0919/weighttest3.txt', 'a')
            #f.write(">>>>>FOLD:    %d \n" % iter)
            #f.write("Correct FER score:    %d \n" % count)
           # f.close()

    elif label==True and str=='AffectNet':
        X = []
        Y = []

        weigths=[0.8]
        w_result=[]
        X, Y, file_list = image_load(str,iter)
        print("===============", file_list)
        X = X.reshape(-1, 128, 128, 1)
        Y = Y.reshape(-1, 1)
        '''count_emotions1 = count_dataset(Y)
        f = open('result_0919/08_top2_model_use_testData_count_{0}_{1}.txt'.format((str), (iter)), 'w')
        f.write('{0} Dataset Test \n'.format((str)))
        f.write('Angry:    %d \n' % count_emotions1[0])
        f.write('Disgust:  %d \n' % count_emotions1[1])
        f.write('Fear:     %d \n' % count_emotions1[2])
        f.write('Happy:    %d \n' % count_emotions1[3])
        f.write('Sad:      %d \n' % count_emotions1[4])
        f.write('Surprise: %d \n' % count_emotions1[5])
        f.write('total:    %d \n' % sum(count_emotions1))
        f.close()'''

        print('Appearance Model Loading...')
        app_model = app_model_load(str, iter)
        print('Appearance Model Loading Complete')
        print('Auto Model Loading...')
        set_keras_backend('theano')
        K.set_image_dim_ordering('th')
        automodel = Autoencoder('Autoencoder/model/weights_best_169_0.03.hdf5')
        print('Auto Model Loading Complete')
        set_keras_backend('tensorflow')
        K.set_image_dim_ordering('tf')
        time_count = 0
        for w in range(len(weigths)):
            count = 0
            Pred = np.zeros(len(Y))
            for i in range(len(file_list[0])):
                print('---------->> Predict Image <<----------')
                print('>> File path : ',file_list[0][i])
                print('>> File label: ',Y[i],'\n')
                ###predict time start
                start_t = time.time()
                appearance_result_list, top2_list = appearance_cnn(app_model, X[i], Y[i])
                print('Appearance Feature Extraction Complete')
                geometric_result_list = geometric_cnn(str, file_list[0][i], iter, top2_list[0], top2_list[1],automodel)
                print('Geometric Feature Extraction Complete')
                ###predict time end
                end_t = time.time()
                time_count += (end_t - start_t)
                if geometric_result_list is not -1:
                    combine_result = joint_feature(appearance_result_list, top2_list, geometric_result_list,weigths[w])
                    print('Combine Feature Extraction Complete')
                else:
                    combine_result = appearance_result_list
                print('Per image time:', float(time_count / len(Y)))
                topk = np.argsort(combine_result)
                top1 = int(topk[5])
                Pred[i] = top1
                print('Emotion {0}: '.format(i), top1)
                print('Emotion {0} real:'.format(i), Y[i])
                print('----------------------------------------')
                if int(Y[i]) == int(top1):
                    count += 1

            ### ---------------
            # CONFUSION MATRIX
            ### ---------------
            #draw_confusion_matrix(Y, Pred, str + '08top2model', iter)

    elif label == False:
        X = image_load(path)
        app_model = app_model_load()
        appearance_result_list, top2_list = appearance_cnn(app_model, X, Y)
        geometric_result_list = geometric_cnn(str, i, top2_list[0], top2_list[1])
        combine_result = joint_feature(appearance_result_list, top2_list, geometric_result_list)

        topk = np.argsort(combine_result)
        top1 = int(topk[6])
        print('Emotion result: ', top1)

def image_load(str,cycle):
    if str == 'new' :
        print('Predict New Image')

    elif str == 'CK+' :
        ###npy로 로딩
        X = np.load('app_cnn_6/test_dataset/{0}/{1}_LBP_X_test_dataset.npy'.format((cycle), (2)))
        Y = np.load('app_cnn_6/test_dataset/{0}/{1}_Y_test_dataset.npy'.format((cycle), (2)))
        print('X SHAPE:',X.shape)
        print('Y SHAPE:',Y.shape)

        #X, Y = shuffle(X, Y)
        print(len(X), len(Y))
        print('--->     Loading ck+ Image Data      <---')
        return X, Y

    elif str == 'JAFFE' :
        ###npy로 로딩
        X = np.load('app_cnn_6/test_dataset/{0}/{1}_LBP_X_test_dataset.npy'.format((cycle), (1)))
        Y = np.load('app_cnn_6/test_dataset/{0}/{1}_Y_test_dataset.npy'.format((cycle), (1)))
        print(len(X), len(Y))
        print('--->     Loading jaffe Image Data      <---')
        return X, Y

    elif str == 'FERG' :
        X, Y = ferg.getImageDataFERG_LBP()
        print(len(X), len(Y))
        print('--->     Loading ferg Image Data      <---')
        return X, Y

    elif str == 'AffectNet' :
        X = np.load('app_cnn_6/test_dataset/{0}/{1}_LBP_X_test_dataset.npy'.format((cycle), (3)))
        Y = np.load('app_cnn_6/test_dataset/{0}/{1}_Y_test_dataset.npy'.format((cycle), (3)))
        file_list = np.load('app_cnn_6/test_dataset/{0}/{1}_path_test_dataset.npy'.format((cycle), (3)))
        #X, Y, file_list = affect.loading_Manually_Annotated_file_lists_training('LBP')
        print('--->     Loading AffectNet LBP Image Data      <---')
        return X, Y, file_list

    elif str == 'ALL100' :
        X1, Y1 = ck.getImageDatackplus_LBP()
        #count1 = count_dataset(Y1)
        X1, Y1 = get_n_dataset(X1, Y1)
        X2, Y2 = jaffe.getImageDataJaffe_LBP()
        #count2 = count_dataset(Y2)
        X2, Y2 = get_n_dataset(X2, Y2)
        X3, Y3 = ferg.getImageDataFERG_LBP()
        X3, Y3 = get_n_dataset(X3, Y3)
        X4, Y4, file_list = affect.loading_Manually_Annotated_file_lists_training('LBP')
        print('Predict All db Image')
        return X1, Y1, X2, Y2, X3, Y3, X4, Y4, file_list
    else:
        print('error')

def app_model_load(str, iter):
    ### model load
    if str=='CK+':
        list_of_files = glob.glob('app_cnn_6/model/{0}/CK_*.hdf5'.format((iter)))
        latest_file = max(list_of_files, key=os.path.getctime)
        model = CNN(latest_file)
    elif str=='JAFFE':
        list_of_files = glob.glob('app_cnn_6/model/{0}/JAFFE*.hdf5'.format((iter)))
        latest_file = max(list_of_files, key=os.path.getctime)
        model = CNN(latest_file)
    elif str=='FERG':
        list_of_files = glob.glob('app_cnn_6/model/{0}/FERG*.hdf5'.format((iter)))
        latest_file = max(list_of_files, key=os.path.getctime)
        model = CNN(latest_file)
    elif str=='AffectNet':
        list_of_files = glob.glob('app_cnn_6/model/{0}/AFFECT*.hdf5'.format((iter)))
        latest_file = max(list_of_files, key=os.path.getctime)
        model = CNN(latest_file)

    return model

def appearance_cnn(model,X, Y):
    X= np.expand_dims(X, axis=0)
    get_softmax_result = model.predict(X)
    topk = np.argsort(get_softmax_result[0])
    max1=int(topk[5])
    max2=int(topk[4])
    top2_list=[]
    top2_list.append(max1)
    top2_list.append(max2)
    #print('get_softmax_result[0] sum',np.sum(get_softmax_result[0]))

    return get_softmax_result[0], top2_list

def image_load_geo(str,file_path_idx,automodel):
    X=[]
    Y=[]
    if str == 'new' :
        print('Predict New Image')

    elif str == 'CK+' :
        dirname_ck = 'ckplus/extended-cohn-kanade-images/cohn-kanade-images/'
        X = []
        Y = []
        X = np.load('app_cnn_6/train_dataset/all_path_2_dataset.npy')
        Y = np.load('app_cnn_6/train_dataset/all_Y_2_dataset.npy')
        for i in range(len(X)):
            if os.path.basename(X[i])[:15] == os.path.basename(X[file_path_idx])[:15] and Y[i] == 6:
                diff_g_vec = geo.extract_geometry(X[file_path_idx], X[i])
                break
        return diff_g_vec

    elif str == 'JAFFE' :
        X = []
        Y = []
        X = np.load('app_cnn_6/train_dataset/all_path_1_dataset.npy')
        Y = np.load('app_cnn_6/train_dataset/all_Y_1_dataset.npy')
        for i in range(len(X)):
            if os.path.basename(X[i])[:2] == os.path.basename(X[file_path_idx])[:2] and Y[i] == 6:
                diff_g_vec = geo.extract_geometry(X[file_path_idx], X[i])
                break
        return diff_g_vec

    elif str == 'FERG' :
        X, Y = get_db.get_ferg_path(X, Y)
        for i in range(len(X)):
            dir1, dir2 = os.path.split(os.path.dirname(X[file_path_idx]))
            dir3, name = os.path.split(dir1)

            dir4, dir5 = os.path.split(os.path.dirname(X[i]))
            dir6, search = os.path.split(dir4)

            if search == name and Y[i] == 6:
                print('get_geo file:', X[file_path_idx], '+', i, "번째")
                diff_g_vec = geo.extract_geometry(X[file_path_idx], X[i])
                break
        return diff_g_vec

    elif str == 'AffectNet' :
        set_keras_backend('theano')
        K.set_image_dim_ordering('th')
        neutral_image=auto_main.get_neutral_image(file_path_idx,automodel)
        set_keras_backend('tensorflow')
        K.set_image_dim_ordering('tf')
        diff_g_vec = geo.extract_geometry(file_path_idx,neutral_image,False)
        print('Predict AffectNet Image')
        return diff_g_vec

    elif str == 'ALL100' :
        print('Predict All db Image')
    else:
        print('error')

def geometric_cnn(str, file_path_idx, cycle, emo_1,emo_2,automodel):
    print('emo1, emo2: ', emo_1, '+', emo_2)

    if int(emo_1) > int(emo_2):
        tmp = emo_1
        emo_1 = emo_2
        emo_2 = tmp

    if str=='CK+':
        list_of_files = glob.glob('geo_cnn_6/ck_{0}_fold_{1}_{2}_pair_cnn_model'.format((cycle), (emo_1), (emo_2)) + "*.hdf5")
        filepath = max(list_of_files, key=os.path.getctime)
        print("loading:", filepath)
    elif str=='JAFFE':
        list_of_files = glob.glob('geo_cnn_6/jaffe_{0}_fold_{1}_{2}_pair_cnn_model'.format((cycle), (emo_1), (emo_2)) + "*.hdf5")
        filepath = max(list_of_files, key=os.path.getctime)
        print("loading:", filepath)
    elif str=='FERG':
        list_of_files = glob.glob('geo_cnn_6/ferg_{0}_fold_{1}_{2}_pair_cnn_model'.format((cycle), (emo_1), (emo_2)) + "*.hdf5")
        filepath = max(list_of_files, key=os.path.getctime)
        print("loading:", filepath)
    elif str=='AffectNet':
        list_of_files = glob.glob('geo_cnn_6/affect_{0}_fold_{1}_{2}_pair_cnn_model'.format((cycle), (emo_1), (emo_2)) + "*.hdf5")
        filepath = max(list_of_files, key=os.path.getctime)
        print("loading:", filepath)
    # filepath = "geo_cnn_6/ck_{0}_fold_{1}_{2}_pair_cnn_model".format((1), (emo_1), (emo_2)) + "-weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
    if os.path.exists(filepath):
        model = shallow_CNN(filepath)
        geo_vec = []
        geo_vec = image_load_geo(str, file_path_idx,automodel)
        if geo_vec is not -1:
            geo_vec = geo_vec.reshape(-1, 36, 1, 1)

            get_softmax_result_geo = model.predict(geo_vec)
            return get_softmax_result_geo[0]
        else:
            return -1
    else:
        print('No exist geometric model file')
        return -1

def joint_feature(appearance_result_list, top2_list,geometric_result_list,weight):
    geo_top1=0.0
    geo_top2=0.0
    if int(top2_list[0])>int(top2_list[1]):
        geo_top1=geometric_result_list[1]
        geo_top2=geometric_result_list[0]
    else:
        geo_top1 = geometric_result_list[0]
        geo_top2 = geometric_result_list[1]

    print('--->>      Joint Features      <<---')
    print('Value of appearance_feature: ',appearance_result_list[top2_list[0]],' + ',appearance_result_list[top2_list[1]])
    print('Value of geometric_feature: ', geo_top1, ' + ', geo_top2)

    norm_app_top1=appearance_result_list[top2_list[0]]/(appearance_result_list[top2_list[0]]+appearance_result_list[top2_list[1]])
    norm_app_top2=appearance_result_list[top2_list[1]]/(appearance_result_list[top2_list[0]]+appearance_result_list[top2_list[1]])
    norm_geo_top1=geo_top1/(geo_top1+geo_top2)
    norm_geo_top2=geo_top2/(geo_top1+geo_top2)

    print('Value of normalized appearance_feature: ', norm_app_top1, ' + ', norm_app_top2)
    print('Value of normalized geometric_feature: ', norm_geo_top1, ' + ', norm_geo_top2)

    W=weight
    sum_top1=W*norm_app_top1+(1-W)*norm_geo_top1
    sum_top2=W*norm_app_top2+(1-W)*norm_geo_top2

    print('Value of sum feature: ', sum_top1, ' + ', sum_top2)

    rescaling_top1=(appearance_result_list[top2_list[0]]+appearance_result_list[top2_list[1]])*sum_top1
    rescaling_top2=(appearance_result_list[top2_list[0]]+appearance_result_list[top2_list[1]])*sum_top2

    print('Value of rescaling feature: ', rescaling_top1, ' + ', rescaling_top2)

    appearance_result_list[top2_list[0]]=rescaling_top1
    appearance_result_list[top2_list[1]]=rescaling_top2

    print('Value of rescaling feature sum: ', np.sum(appearance_result_list))
    print('----------------------------------------')
    return appearance_result_list

if __name__=="__main__":
    print('-----------------------------------------')
    print('>>          FER DEMO SYSTEM            <<')
    print('-----------------------------------------')
    print('-----------------------------------------')
    print('> (1) : New Image')
    print('> (2) : CK+')
    print('> (3) : JAFFE')
    print('> (4) : FERG')
    print('> (5) : AffectNet')
    print('> (6) : ALL100')
    print('> (7) : EXIT')
    print('-----------------------------------------\n')
    str = input('> 작업 선택 : ')

    if str == '1' :
        fer_main('new')
    elif str == '2' :
        fer_main('CK+')
    elif str == '3' :
        fer_main('JAFFE')
    elif str == '4' :
        fer_main('FERG')
    elif str == '5' :
        fer_main('AffectNet')
    elif str == '6' :
        fer_main('ALL100')
    elif str == '7' :
        exit(-1)
    else:
        print('\n잘못된 입력')
        exit(-1)

    K.clear_session()
    exit(0)