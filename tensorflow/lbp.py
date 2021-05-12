# -*- coding: utf-8 -*-
"""
Created on Mon Jan 22 13:09:37 2018

@author: KIMJIHAE
"""

import cv2
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import dlib
from matplotlib import pyplot as plt
from randmark_detect import rect_to_bb, shape_to_np, get_left_eye, get_right_eye, get_nose, get_mouse, get_img

def get_pixel(img, center, x, y):#중심픽셀과 비교해서 크면 1 아니면 0으로 만드는 함수
    new_value = 0
    try:
        if img[x][y] >= center:
            new_value = 1
    except:
        pass
    return new_value

def lbp_calculated_pixel(img, x, y):
    '''
     64 | 128 |   1
    ----------------
     32 |   0 |   2
    ----------------
     16 |   8 |   4    
    '''    
    center = img[x][y]
    val_ar = []   #위 블록 모양처럼 차례대로 1010...붙임
    val_ar.append(get_pixel(img, center, x-1, y+1))     # top_right
    val_ar.append(get_pixel(img, center, x, y+1))       # right
    val_ar.append(get_pixel(img, center, x+1, y+1))     # bottom_right
    val_ar.append(get_pixel(img, center, x+1, y))       # bottom
    val_ar.append(get_pixel(img, center, x+1, y-1))     # bottom_left
    val_ar.append(get_pixel(img, center, x, y-1))       # left
    val_ar.append(get_pixel(img, center, x-1, y-1))     # top_left
    val_ar.append(get_pixel(img, center, x-1, y))       # top
    
    power_val = [1, 2, 4, 8, 16, 32, 64, 128]
    val = 0
    for i in range(len(val_ar)): #위에서 이진 배열로 나타낸 것을 10진수로 만들어 리턴
        val += val_ar[i] * power_val[i]
    return val

def show_output(output_list): #lbp 결과 그래프와 이미지로 보여줌.
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
        current_plot = figure.add_subplot(1, output_list_len, i+1)
        if current_type == "gray":
            current_plot.imshow(current_img, cmap = plt.get_cmap('gray'))
            current_plot.set_title(current_title)
            current_plot.set_xticks(current_xtick)
            current_plot.set_yticks(current_ytick)
            current_plot.set_xlabel(current_xlabel)
            current_plot.set_ylabel(current_ylabel)
        elif current_type == "histogram":
            current_plot.plot(current_img, color = "black")
            current_plot.set_xlim([0,260]) #0~255까지의 숫자를 표현
            current_plot.set_title(current_title)
            current_plot.set_xlabel(current_xlabel)
            current_plot.set_ylabel(current_ylabel)            
            ytick_list = [int(i) for i in current_plot.get_yticks()]
            current_plot.set_yticklabels(ytick_list,rotation = 90)

    plt.show()
    
def main():
    image_file = 'FERG_DB_256/aia/aia_joy/aia_joy_1.png' #이미지 가지고 오는 부분
    img_bgr = cv2.imread(image_file)
    height, width, channel = img_bgr.shape
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    
    img_lbp = np.zeros((height, width,3), np.uint8)
    for i in range(0, height):
        for j in range(0, width):
             img_lbp[i, j] = lbp_calculated_pixel(img_gray, i, j) #각 픽셀마다 lbp 계산해서 배열에 입력.
    hist_lbp = cv2.calcHist([img_lbp], [0], None, [256], [0, 256]) #opencv lib 이용해서 히스토그램그림
    
    hist,bins = np.histogram(img_lbp.flatten(),256,[0,256])
    cdf = hist.cumsum()
    cdf_normalized = cdf * hist.max()/ cdf.max()
    plt.plot(cdf_normalized, color = 'b')
    plt.hist(img_lbp.flatten(),256,[0,256], color = 'r')
    plt.xlim([0,256])
    plt.legend(('cdf','histogram'), loc = 'upper left')
    plt.show()
    
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
    output_list.append({
        "img": img_lbp,
        "xlabel": "",
        "ylabel": "",
        "xtick": [],
        "ytick": [],
        "title": "LBP Image",
        "type": "gray"
    })    
    output_list.append({
        "img": hist_lbp,
        "xlabel": "Bins",
        "ylabel": "Number of pixels",
        "xtick": None,
        "ytick": None,
        "title": "Histogram(LBP)",
        "type": "histogram"
    })

    show_output(output_list)
                             
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print("LBP Program is finished")
    
def main_concat9():
    image_file = 'FERG_DB_256/aia/aia_joy/Resize_aia_joy_1.png' #이미지 가지고 오는 부분
    img_bgr = cv2.imread(image_file)
    height, width, channel = img_bgr.shape
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    
    img_lbp = np.zeros((height, width), np.uint16)
    img_lbp_block = np.zeros((height, width), np.uint16)

    #hist_lbp = []
    for i in range(0, height):
        for j in range(0, width):
             img_lbp[i, j] = lbp_calculated_pixel(img_gray, i, j)
             
    for y in range(0,4):
        for x in range(0,4):
            sh = 0+int(height/4)*y
            fh = int(height/4)+int(height/4)*y
            sw = 0+int(width/4)*x
            fw = int(width/4)+int(width/4)*x
            for h in range(sh, fh):
                for w in range(sw, fw):
                    #img_lbp_block[(j+(i*4)),h-int(height/4)*i, x-int(width/4)*j] = img_lbp[h, x]
                    img_lbp_block[h, w] = img_lbp[h, w] + 255*(x+(y*4))
                    #print(histplus)
                    #print('x: ',w,'y: ',h,'val: ',img_lbp[h, w])
    
    #hist_lbp = cv2.calcHist([img_lbp_block], [0], None,256*15,[0,256*15])

    hist,bins = np.histogram(img_lbp_block.flatten(),(256*15),[0,(256*15)])
    cdf = hist.cumsum()
    cdf_normalized = cdf * hist.max()/ cdf.max()
    plt.plot(cdf_normalized, color = 'b')
    plt.hist(img_lbp_block.flatten(),(256*15),[0,(256*15)], color = 'r')
    plt.xlim([0,(256*15)])
    plt.legend(('cdf','histogram'), loc = 'upper left')
    plt.show()
    
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
    output_list.append({
        "img": img_lbp,
        "xlabel": "",
        "ylabel": "",
        "xtick": [],
        "ytick": [],
        "title": "LBP Image",
        "type": "gray"
    })    
    '''output_list.append({
        "img": hist_lbp,
        "xlabel": "Bins",
        "ylabel": "Number of pixels",
        "xtick": None,
        "ytick": None,
        "title": "Histogram(LBP)",
        "type": "histogram"
    })'''

    show_output(output_list)
                             
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print("LBP Program is finished")

def neutral_roi():
    image_file = 'FERG_DB_256/mery/mery_neutral/mery_neutral_1.png' #이미지 가지고 오는 부분
    ''''''
    
def img_load():
    #image_file = 'jaffe/AN_resize037210.png'
    image_file = 'FERG_DB_256/aia/aia_disgust/Resize_aia_disgust_1.png' #이미지 가지고 오는 부분
    #image_file = 'ckplus/extended-cohn-kanade-images/cohn-kanade-images/S022/005/Resize_S022_005_00000032.png'
    img_bgr = cv2.imread(image_file)
    height, width, channel = img_bgr.shape
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    
    img_gray = cv2.equalizeHist(img_gray) #히스토그램 평활화
    
    img_lbp = np.zeros((height, width), np.uint16)
    
    return image_file, height, width, img_gray, img_lbp

def make_roi_lbp_array(roi, img_lbp, idx):
    roi_lbp_block = np.zeros((roi[3]-roi[1], roi[2]-roi[0]), np.uint16)
    
    for y in range(roi[1],roi[3]-1):
        for x in range(roi[0],roi[2]-1):
            roi_lbp_block[y-roi[1], x-roi[0]] = img_lbp[y, x] + (255*idx)
    
    return roi_lbp_block

def make_histogram(img_gray,img_lbp,concat_block, left_eye_lbp_block):
    hist,bins = np.histogram(concat_block.flatten(),(256*3),[0,(256*3)])
    cdf = hist.cumsum()
    cdf_normalized = cdf * hist.max()/ cdf.max()
    plt.plot(cdf_normalized, color = 'b')
    plt.hist(concat_block.flatten(),(256*3),[0,(256*3)], color = 'r')
    plt.xlim([0,(256*3)])
    plt.legend(('cdf','histogram'), loc = 'upper left')
    plt.show()
    
    #hist = hist-neutral_roi()
    min_max_scaler = MinMaxScaler()
    concat_norm = min_max_scaler.fit_transform(hist)
    plt.plot(softmax(concat_norm), color = 'b')
    plt.legend(('histogram_softmax'), loc = 'upper left')
    plt.show()

    
def make_image_show(img_gray,img_lbp,left_eye_lbp_block,right_eye_lbp_block,nose_lbp_block,mouse_lbp_block):
    output_list = []
    output_list2 = []
    output_list.append({
        "img": img_gray,
        "xlabel": "",
        "ylabel": "",
        "xtick": [],
        "ytick": [],
        "title": "Gray Image",
        "type": "gray"        
    })
    output_list.append({
        "img": img_lbp,
        "xlabel": "",
        "ylabel": "",
        "xtick": [],
        "ytick": [],
        "title": "LBP Image",
        "type": "gray"
    })
    output_list2.append({
        "img": left_eye_lbp_block,
        "xlabel": "",
        "ylabel": "",
        "xtick": [],
        "ytick": [],
        "title": "left_eye",
        "type": "gray"        
    })
    output_list2.append({
        "img": right_eye_lbp_block,
        "xlabel": "",
        "ylabel": "",
        "xtick": [],
        "ytick": [],
        "title": "right_eye",
        "type": "gray"        
    })
    output_list2.append({
        "img": nose_lbp_block,
        "xlabel": "",
        "ylabel": "",
        "xtick": [],
        "ytick": [],
        "title": "nose",
        "type": "gray"        
    })
    output_list2.append({
        "img": mouse_lbp_block,
        "xlabel": "",
        "ylabel": "",
        "xtick": [],
        "ytick": [],
        "title": "mouse",
        "type": "gray"        
    })
    
    show_output(output_list)
    show_output(output_list2)

def main_roi():
    image_file, height, width, img_gray, img_lbp = img_load()
    
    predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
    image, gray, rects = get_img(image_file)
    shape = predictor(gray, rects[0])
    shape = shape_to_np(shape)
    
    left_eye = get_left_eye(shape)
    right_eye = get_right_eye(shape)
    nose = get_nose(shape)
    mouse = get_mouse(shape)
    
    #hist_lbp = []
    for i in range(0, height):
        for j in range(0, width):
             img_lbp[i, j] = lbp_calculated_pixel(img_gray, i, j)
    #img_lbp=abs(img_lbp-neutral_roi())
    
    left_eye_lbp_block = make_roi_lbp_array(left_eye,img_lbp, 0)
    right_eye_lbp_block = make_roi_lbp_array(right_eye,img_lbp, 1)
    nose_lbp_block = make_roi_lbp_array(nose,img_lbp, 2)
    mouse_lbp_block = make_roi_lbp_array(mouse,img_lbp, 3)
    
    concat_block = np.concatenate((left_eye_lbp_block.flatten(),right_eye_lbp_block.flatten(),nose_lbp_block.flatten(),mouse_lbp_block.flatten()))
    
    make_histogram(img_gray,img_lbp,concat_block, left_eye_lbp_block)
        
    make_image_show(img_gray,img_lbp,left_eye_lbp_block,right_eye_lbp_block,nose_lbp_block,mouse_lbp_block)
                             
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print("LBP Program is finished")

def softmax(w, t = 1.0):
    e = np.exp(np.array(w))
    dist = e / np.sum(e,axis=0)
    min_max_scaler = MinMaxScaler()
    dist = min_max_scaler.fit_transform(dist)
    #dist = e / np.sum(e,axis=0)
    return dist

def softmax_2(x):
    e_x = np.exp(x - np.max(x))
    out = e_x / e_x.sum()
    return out

if __name__ == '__main__':
    #w = np.array([-0.1,0.2])
    #print(softmax(w))
    main_roi()