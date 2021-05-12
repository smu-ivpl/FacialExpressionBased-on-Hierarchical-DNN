# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 15:35:59 2018

@author: KIMJIHAE
"""

# import the necessary packages
import numpy as np
import imutils
import dlib
import cv2
import os

def rect_to_bb(rect):
    # take a bounding predicted by dlib and convert it
    # to the format (x, y, w, h) as we would normally do
    # with OpenCV
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y
 
    # return a tuple of (x, y, w, h)
    return (x, y, w, h)

def shape_to_np(shape, dtype="int"):
    # initialize the list of (x, y)-coordinates
    coords = np.zeros((68, 2), dtype=dtype)
 
    # loop over the 68 facial landmarks and convert them
    # to a 2-tuple of (x, y)-coordinates
    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)
 
    # return the list of (x, y)-coordinates
    return coords


def get_left_eye(coords):
    # initialize the list of (x, y)-coordinates
    left_eye = np.zeros(4, dtype=int)
    left_eye = (coords[17][0],coords[19][1],coords[21][0],coords[29][1])
    
    return left_eye

def get_right_eye(coords):
    # initialize the list of (x, y)-coordinates
    right_eye = np.zeros(4, dtype=int)
    right_eye = (coords[22][0],coords[24][1],coords[26][0],coords[29][1])
    
    return right_eye

def get_nose(coords):
    # initialize the list of (x, y)-coordinates
    nose = np.zeros(4, dtype=int)
    nose = (int((coords[40][0]+coords[41][0])/2),int((coords[19][1]+coords[24][1])/2),int((coords[46][0]+coords[47][0])/2),coords[33][1])
    
    return nose

def get_mouse(coords):
    # initialize the list of (x, y)-coordinates
    mouse = np.zeros(4, dtype=int)
    mouse = (coords[48][0],int((coords[50][1]+coords[52][1])/2),coords[54][0],coords[8][1])
    
    return mouse

def get_img(args):
    detector = dlib.get_frontal_face_detector()
    # load the input image, resize it, and convert it to grayscale
    image = cv2.imread(args)
    #image = imutils.resize(image, width=500)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
     
    # detect faces in the grayscale image
    rects = detector(gray, 1)
    #rects = tuple(np.array(rects[0]).reshape(1,4))
    #rects = rect_to_bb(rects[0])
    print('rects tuple:',rects)
    dlib_rect = [(rects[0].left(), rects[0].top(), rects[0].right() - rects[0].left(), rects[0].bottom() - rects[0].top())]
    #dlib_rect = dlib.rectangle(left=rects[0].left(),top=rects[0].top(),right=rects[0].right(),bottom=rects[0].bottom())
    return image, gray, dlib_rect

def main():
    args = os.path.join('jaffe/', "AN_resize037210.png")
    # initialize dlib's face detector (HOG-based) and then create
    # the facial landmark predictor
    predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
    
    image, gray, rects = get_img('FERG_DB_256/aia/aia_disgust/Resize_aia_disgust_1.png')
    
    # loop over the face detections
    for (i, rect) in enumerate(rects):
        # determine the facial landmarks for the face region, then
        # convert the facial landmark (x, y)-coordinates to a NumPy
        # array
        shape = predictor(gray, rect)
        shape = shape_to_np(shape)
        
        left_eye = get_left_eye(shape)
        right_eye = get_right_eye(shape)
        nose = get_nose(shape)
        mouse = get_mouse(shape)
        
        # convert dlib's rectangle to a OpenCV-style bounding box
        # [i.e., (x, y, w, h)], then draw the face bounding box
        (x, y, w, h) = rect_to_bb(rect)
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
     
        # show the face number
        cv2.putText(image, "Face #{}".format(i + 1), (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
     
        # loop over the (x, y)-coordinates for the facial landmarks
        # and draw them on the image
        for (x, y) in shape:
            cv2.circle(image, (x, y), 1, (0, 0, 255), -1)
            
        cv2.rectangle(image, (left_eye[0], left_eye[1]), (left_eye[2], left_eye[3]), (0, 255, 255), 2)
        cv2.rectangle(image, (right_eye[0], right_eye[1]), (right_eye[2], right_eye[3]), (0, 255, 255), 2)
        cv2.rectangle(image, (nose[0], nose[1]), (nose[2], nose[3]), (0, 255, 255), 2)
        cv2.rectangle(image, (mouse[0], mouse[1]), (mouse[2], mouse[3]), (0, 255, 255), 2)
        
    # show the output image with the face detections + facial landmarks
    cv2.imshow("Output", image)
    cv2.waitKey(0)

if __name__ == '__main__':
    main()