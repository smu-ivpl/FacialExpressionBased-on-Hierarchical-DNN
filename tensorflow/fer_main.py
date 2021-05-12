import numpy as np
from keras.utils import np_utils
from cnn_keras import CNN
from keras.models import load_model
import itertools
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
from train_cnn import loading_n_data, get_n_dataset, count_dataset, draw_confusion_matrix
from keras import backend as K
from pylab import *
import os
from os import environ
from crop_images import faceImageCrop

def fer_main(str):
    if str == 'new' :
        print('Predict New Image')

    elif str == 'CK+' :
        label = True
        result_emotion(str, label)
        print('Predict CK+ Image')

    elif str == 'JAFFE' :
        print('Predict JAFFE Image')
        label=True
        result_emotion(str, label)

    elif str == 'FERG' :
        label = True
        result_emotion(str, label)
        print('Predict FERG Image')

    elif str == 'AffectNet' :
        label = True
        result_emotion(str, label)
        print('Predict AffectNet Image')

    elif str == 'ALL100' :
        label = True
        result_emotion_all(str)
        print('Predict All db Image')
    else:
        print('error')

def result_emotion_all(str):
    X1, Y1, X2, Y2, X3, Y3, X4, Y4, file_list = image_load(str)
    count1 = 0
    count2 = 0
    count3 = 0
    count4 = 0
    Y_test=[]
    predict=[]
    print('Appearance Model Loading...')
    app_model = app_model_load()
    print('Appearance Model Loading Complete')

    ### CK+ ---------------------------------------------------------------------------------------
    for i in range(len(Y1)):
        appearance_result_list, top2_list = appearance_cnn(app_model, X1[i], Y1[i])
        print('Appearance Feature Extraction Complete')
        geometric_result_list = geometric_cnn('CK+', i, top2_list[0], top2_list[1])
        print('Geometric Feature Extraction Complete')
        if geometric_result_list is not -1:
            combine_result = joint_feature(appearance_result_list, top2_list, geometric_result_list)
            print('Combine Feature Extraction Complete')
        else:
            combine_result = appearance_result_list
            print('Only using Appearance Feature')
        topk = np.argsort(combine_result)
        top1 = int(topk[6])
        print('Emotion {0} predict: '.format(i), top1)
        print('Emotion {0} real:'.format(i), Y1[i])
        Y_test.append(top1)
        predict.append(Y1[i])
        if int(Y1[i]) == int(top1):
            count1 += 1
    print('Correct ck+ FER score (%): ',
          ((count1) / (len(Y1))) * 100)
    # ---------------------------------------------------------------------------------------------
    ### JAFFE ---------------------------------------------------------------------------------------
    i=0
    for i in range(len(Y2)):
        appearance_result_list, top2_list = appearance_cnn(app_model, X2[i], Y2[i])
        print('Appearance Feature Extraction Complete')
        geometric_result_list = geometric_cnn('JAFFE', i, top2_list[0], top2_list[1])
        print('Geometric Feature Extraction Complete')
        if geometric_result_list is not -1:
            combine_result = joint_feature(appearance_result_list, top2_list, geometric_result_list)
            print('Combine Feature Extraction Complete')
        else:
            combine_result = appearance_result_list
            print('Only using Appearance Feature')
        topk = np.argsort(combine_result)
        top1 = int(topk[6])
        print('Emotion {0} predict: '.format(i), top1)
        print('Emotion {0} real:'.format(i), Y2[i])
        Y_test.append(top1)
        predict.append(Y2[i])
        if int(Y2[i]) == int(top1):
            count2 += 1
    # ---------------------------------------------------------------------------------------------
    print('Correct jaffe FER score (%): ',
          ((count2) / (len(Y2))) * 100)
    ### FERG ---------------------------------------------------------------------------------------
    i = 0
    for i in range(len(Y3)):
        appearance_result_list, top2_list = appearance_cnn(app_model, X3[i], Y3[i])
        print('Appearance Feature Extraction Complete')
        geometric_result_list = geometric_cnn('FERG', i, top2_list[0], top2_list[1])
        print('Geometric Feature Extraction Complete')
        if geometric_result_list is not -1:
            combine_result = joint_feature(appearance_result_list, top2_list, geometric_result_list)
            print('Combine Feature Extraction Complete')
        else:
            combine_result = appearance_result_list
            print('Only using Appearance Feature')
        topk = np.argsort(combine_result)
        top1 = int(topk[6])
        print('Emotion {0} predict: '.format(i), top1)
        print('Emotion {0} real:'.format(i), Y3[i])
        Y_test.append(top1)
        predict.append(Y3[i])
        if int(Y3[i]) == int(top1):
            count3 += 1
    # ---------------------------------------------------------------------------------------------


    print('Correct ferg FER score (%): ',
          ((count3) / (len(Y3))) * 100)
    ### AffectNet ---------------------------------------------------------------------------------------
    i = 0
    for i in range(len(file_list)):
        print('---------->> Predict Image <<----------')
        print('>> File path : ',file_list[i])
        print('>> File label: ',Y4[i],'\n')
        appearance_result_list, top2_list = appearance_cnn(app_model, X4[i], Y4[i])
        print('Appearance Feature Extraction Complete')
        geometric_result_list = geometric_cnn('AffectNet', file_list[i], top2_list[0], top2_list[1])
        print('Geometric Feature Extraction Complete')
        if geometric_result_list is not -1:
            combine_result = joint_feature(appearance_result_list, top2_list, geometric_result_list)
            print('Combine Feature Extraction Complete')
        else:
            combine_result = appearance_result_list

        topk = np.argsort(combine_result)
        top1 = int(topk[6])
        print('Predict Emotion {0}: '.format(i), top1)
        print('----------------------------------------')
        Y_test.append(top1)
        predict.append(Y4[i])
        if int(Y4[i]) == int(top1):
            count4 += 1
    # ---------------------------------------------------------------------------------------------
    print('Correct ck+ FER score (%): ',
          ((count1) / (len(Y1))) * 100)
    print('Correct jaffe FER score (%): ',
          ((count2) / (len(Y2))) * 100)
    print('Correct ferg FER score (%): ',
          ((count3) / (len(Y3))) * 100)
    print('Correct affectnet FER score (%): ',
          ((count4) / (len(file_list))) * 100)
    print('Correct all FER score (%): ', ((count1+count2+count3+count4) / (len(file_list)+len(Y1)+len(Y2)+len(Y3))) * 100)
    draw_confusion_matrix(Y_test, predict, 'All_dataset')

def result_emotion(str, label=True):
    if label==True and (str=='CK+' or str=='JAFFE' or str=='FERG'):
        X = []
        Y = []
        w = 0.7
        X, Y = image_load(str)
        Pred = np.zeros(len(Y))
        count_emotions1 = count_dataset(Y)
        f = open('results/top2_model_use_testData_count_{0}.txt'.format((str)), 'w')
        f.write('{0} Dataset Test \n'.format((str)))
        f.write('Angry:    %d \n' % count_emotions1[0])
        f.write('Disgust:  %d \n' % count_emotions1[1])
        f.write('Fear:     %d \n' % count_emotions1[2])
        f.write('Happy:    %d \n' % count_emotions1[3])
        f.write('Sad:      %d \n' % count_emotions1[4])
        f.write('Surprise: %d \n' % count_emotions1[5])
        f.write('Neutral:  %d \n' % count_emotions1[6])
        f.write('total:    %d \n' % sum(count_emotions1))
        f.close()
        count = 0
        print('Appearance Model Loading...')
        app_model = app_model_load()
        print('Appearance Model Loading Complete')

        for i in range(len(Y)):
            appearance_result_list, top2_list = appearance_cnn(app_model, X[i], Y[i])
            print('Appearance Feature Extraction Complete')
            geometric_result_list = geometric_cnn(str,i, top2_list[0], top2_list[1])
            print('Geometric Feature Extraction Complete')
            if geometric_result_list is not -1:
                combine_result = joint_feature(appearance_result_list, top2_list, geometric_result_list,w)
                print('Combine Feature Extraction Complete')
            else:
                combine_result = appearance_result_list
                print('Only using Appearance Feature')

            topk = np.argsort(combine_result)
            top1 = int(topk[6])
            Pred[i]=top1
            print('Emotion {0}: '.format(i), top1)
            print('Emotion {0} real:'.format(i), Y[i])
            if int(Y[i]) == int(top1):
                count += 1
        ### ---------------
        # CONFUSION MATRIX
        ### ---------------
        draw_confusion_matrix(Y, Pred, str+'top2model')
        print('Save Confusion Matrix')
        print('Correct FER score (%): ', (count / len(Y)) * 100)

    elif label==True and str=='AffectNet':
        X = []
        Y = []

        weigths=[0.7]
        w_result=[]
        X, Y, file_list = image_load(str)
        Pred = np.zeros(len(Y))
        count_emotions=count_dataset(Y)
        count = 0
        print('AffectNet Dataset Test \n')
        print('Angry:    {0} '.format((count_emotions[0])) )
        print('Disgust:  {0} '.format((count_emotions[1])) )
        print('Fear:     {0} '.format((count_emotions[2])) )
        print('Happy:    {0} '.format((count_emotions[3])) )
        print('Sad:      {0} '.format((count_emotions[4])) )
        print('Surprise: {0} '.format((count_emotions[5])) )
        print('Neutral:  {0} '.format((count_emotions[6])) )
        print('total:    {0} '.format((sum(count_emotions))) )

        f = open('results/weight_ratio_result_top3.txt', 'w')
        f.write('AffectNet Dataset Test \n')
        f.write('Angry:    %d \n'% count_emotions[0])
        f.write('Disgust:  %d \n'% count_emotions[1])
        f.write('Fear:     %d \n'% count_emotions[2])
        f.write('Happy:    %d \n'% count_emotions[3])
        f.write('Sad:      %d \n'% count_emotions[4])
        f.write('Surprise: %d \n'% count_emotions[5])
        f.write('Neutral:  %d \n'% count_emotions[6])
        f.write('total:    %d \n'% sum(count_emotions))
        print('Appearance Model Loading...')
        app_model = app_model_load()
        print('Appearance Model Loading Complete')


        for w in range(len(weigths)):
            count = 0
            for i in range(len(file_list)):
                print('---------->> Predict Image <<----------')
                print('>> File path : ',file_list[i])
                print('>> File label: ',Y[i],'\n')
                appearance_result_list, top2_list = appearance_cnn(app_model, X[i], Y[i])
                print('Appearance Feature Extraction Complete')
                geometric_result_list = geometric_cnn(str, file_list[i], top2_list[0], top2_list[1])
                print('Geometric Feature Extraction Complete')
                if geometric_result_list is not -1:
                    combine_result = joint_feature(appearance_result_list, top2_list, geometric_result_list,weigths[w])
                    print('Combine Feature Extraction Complete')
                else:
                    combine_result = appearance_result_list

                topk = np.argsort(combine_result)
                top1 = int(topk[6])
                Pred[i] = top1
                print('Predict Emotion {0}: '.format(i), top1)
                print('----------------------------------------')
                if int(Y[i]) == int(top1):
                    count += 1

                ### ---------------
                # CONFUSION MATRIX
                ### ---------------
            print('Correct FER score (%): ', (count / len(file_list)) * 100)
            count_result= (count / len(file_list)) * 100
            f.write('Correct FER count : %d \n' % count)
            f.write('Correct FER score : %f \n'%count_result)
            w_result.append((count / len(file_list)) * 100)
        f.close()
        draw_confusion_matrix(Y, Pred, str + 'top2model')
        print('Save Confusion Matrix')
        print('Correct FER score (%): ', (count / len(Y)) * 100)
        '''plt.figure()
        plt.plot(w_result)
        plt.title('Top2 Features Join Weight Ratio Graph')
        plt.xlabel('Ratio of appearance feature')
        plt.ylabel('accuracy')
        plt.savefig('results/weigth_graph/weight_ratio_graph_top3.png')'''

    elif label == False:
        X = image_load(path)
        app_model = app_model_load()
        appearance_result_list, top2_list = appearance_cnn(app_model, X, Y)
        geometric_result_list = geometric_cnn(str, i, top2_list[0], top2_list[1])
        combine_result = joint_feature(appearance_result_list, top2_list, geometric_result_list)

        topk = np.argsort(combine_result)
        top1 = int(topk[6])
        print('Emotion result: ', top1)

def image_load(str):
    if str == 'new' :
        print('Predict New Image')

    elif str == 'CK+' :
        #X, Y = ck.getImageDatackplus_LBP()
        X, Y = ck.getImageDatackplus_LBP_flip()
        X1, Y1 = ck.getImageDatackplus_LBP_noise()
        X = np.concatenate((X, X1), axis=0)
        Y = np.concatenate((Y, Y1), axis=0)
        print('X SHAPE:',X.shape)
        print('Y SHAPE:',Y.shape)

        #X, Y = shuffle(X, Y)
        print(len(X), len(Y))
        print('--->     Loading ck+ Image Data      <---')
        return X, Y

    elif str == 'JAFFE' :
        #X, Y = jaffe.getImageDataJaffe_LBP()
        X, Y = jaffe.getImageDataJaffe_LBP_flip()
        X1, Y1 = jaffe.getImageDataJaffe_LBP_noise()
        X = np.concatenate((X, X1), axis=0)
        Y = np.concatenate((Y, Y1), axis=0)
        #X, Y = shuffle(X, Y)
        print(len(X), len(Y))
        print('--->     Loading jaffe Image Data      <---')
        return X, Y

    elif str == 'FERG' :
        X, Y = ferg.getImageDataFERG_LBP()
        print(len(X), len(Y))
        print('--->     Loading ferg Image Data      <---')
        return X, Y

    elif str == 'AffectNet' :
        X, Y, file_list = affect.loading_Manually_Annotated_file_lists_training('LBP')
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

def app_model_load():
    model = CNN("LBP(all)/model/weights_best_49_0.84.hdf5")
    return model

def appearance_cnn(model,X, Y):
    X= np.expand_dims(X, axis=0)
    get_softmax_result = model.predict(X)
    topk = np.argsort(get_softmax_result[0])
    max1=int(topk[6])
    max2=int(topk[5])
    top2_list=[]
    top2_list.append(max1)
    top2_list.append(max2)
    #print('get_softmax_result[0] sum',np.sum(get_softmax_result[0]))

    return get_softmax_result[0], top2_list

def image_load_geo(str,file_path_idx):
    X=[]
    Y=[]
    if str == 'new' :
        print('Predict New Image')

    elif str == 'CK+' :
        dirname_ck = 'ckplus/extended-cohn-kanade-images/cohn-kanade-images/'
        X, Y = get_db.get_ck_plus_path(dirname_ck, X, Y)
        X = np.concatenate((X, X), axis=0)
        Y = np.concatenate((Y, Y), axis=0)
        for i in range(len(X)):
            if os.path.basename(X[i])[:15] == os.path.basename(X[file_path_idx])[:15] and Y[i] == 6:
                diff_g_vec = geo.extract_geometry(X[file_path_idx], X[i])
                break
        return diff_g_vec

    elif str == 'JAFFE' :
        X, Y = get_db.get_jaffe_path(X, Y)
        X = np.concatenate((X, X), axis=0)
        Y = np.concatenate((Y, Y), axis=0)
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
        neutral_image=auto_main.get_neutral_image(file_path_idx)
        diff_g_vec = geo.extract_geometry(file_path_idx,neutral_image,False)
        print('Predict AffectNet Image')
        return diff_g_vec

    elif str == 'ALL100' :
        print('Predict All db Image')
    else:
        print('error')

def geometric_cnn(str, file_path_idx,emo_1,emo_2):
    print('emo1, emo2: ',emo_1,'+',emo_2)

    if int(emo_1)>int(emo_2):
        tmp=emo_1
        emo_1=emo_2
        emo_2=tmp

    if os.path.exists("geo_weight/{0}_{1}_pair_cnn_model.hdf5".format((emo_1), (emo_2))):
        model = shallow_CNN("geo_weight/{0}_{1}_pair_cnn_model.hdf5".format((emo_1), (emo_2)))
        geo_vec=[]
        geo_vec=image_load_geo(str,file_path_idx)
        if geo_vec is not -1:
            geo_vec=geo_vec.reshape(-1,1,36,1)

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