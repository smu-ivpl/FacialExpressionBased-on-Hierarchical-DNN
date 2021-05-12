# -*- coding: utf-8 -*-
"""
Created on Sat Aug 26 16:37:05 2017

@author: kimjihae
"""
#Python 2.7.2
#Opencv 2.4.2
#PIL 1.1.7

#dlib써서 IMAGE CROP
import dlib
from PIL import Image
import matplotlib.pyplot as plt
from skimage import io
import glob
import os
import cv2

def detect_faces(image):

    # Create a face detector
    face_detector = dlib.get_frontal_face_detector()

    # Run detector and get bounding boxes of the faces on image.
    detected_faces = face_detector(image, 1)
    face_frames = [(x.left(), x.top(),
                    x.right(), x.bottom()) for x in detected_faces]

    return face_frames

# Load image
'''img_path = 'test.png'
image = io.imread(img_path)

# Detect faces
detected_faces = detect_faces(image)

# Crop faces and plot
for n, face_rect in enumerate(detected_faces):
    face = Image.fromarray(image).crop(face_rect)
    plt.subplot(1, len(detected_faces), n+1)
    plt.axis('off')
    plt.imshow(face)
'''
'''def faceCrop(imagePattern,boxScale=1):
    imgList=glob.glob(imagePattern)
    if len(imgList)<=0:
        print ('No Images Found')
        return

    for img in imgList:
        pil_im=io.imread(img)# Load image
        faces=detect_faces(pil_im)
        if faces:
            for n, face_rect in enumerate(faces):
                face = Image.fromarray(pil_im).crop(face_rect)
                
                fname,ext=os.path.splitext(img)
                face.save(fname+'_crop'+str(n)+ext)
                plt.subplot(1, len(faces), n+1)
                plt.axis('off')
                plt.imshow(face)
        else:
            print ('No faces found:', img)
    
faceCrop('testPics/*.png',boxScale=1) '''   
    
def faceImageCrop(img, boxScale=1):
    pil_im=io.imread(img)# Load image

    face = Image.fromarray(pil_im)
    #faces=detect_faces(pil_im)
    # Create a face detector
    face_detector = dlib.get_frontal_face_detector()

    image_cv = cv2.imread(img)
    gray_cv = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
    # Run detector and get bounding boxes of the faces on image.
    detected_faces = face_detector(gray_cv, 1)
    faces = [(x.left(), x.top(),
                    x.right(), x.bottom()) for x in detected_faces]

    count=0
    if faces:
        for n, face_rect in enumerate(faces):
            if count>=1:
                break
            else:
                face = Image.fromarray(pil_im).crop(face_rect)
                count+=1
            #fname,ext=os.path.splitext(img)
            #face.save(fname+'_crop'+str(n)+ext)
            #plt.subplot(1, len(faces), n+1)
            #plt.axis('off')
            #plt.imshow(face)
        return face
    else:
        print ('No faces found:', img)
        return -1
    
    #return face


#HARRCASCADE써서 IMAGE CROP
'''import cv
#import cv2 as cv #Opencv
from PIL import Image #Image from PIL
import glob
import os

def DetectFace(image, faceCascade, returnImage=False):
    # This function takes a grey scale cv image and finds
    # the patterns defined in the haarcascade function
    # modified from: http://www.lucaamore.com/?p=638

    #variables    
    min_size = (20,20)
    haar_scale = 1.1
    min_neighbors = 3
    haar_flags = 0
    
    # Equalize the histogram
    cv.EqualizeHist(image, image)

    # Detect the faces
    faces = cv.HaarDetectObjects(
            image, faceCascade, cv.CreateMemStorage(0),
            haar_scale, min_neighbors, haar_flags, min_size
        )

    # If faces are found
    if faces and returnImage:
        for ((x, y, w, h), n) in faces:
            # Convert bounding box to two CvPoints
            pt1 = (int(x), int(y))
            pt2 = (int(x + w), int(y + h))
            cv.Rectangle(image, pt1, pt2, cv.RGB(255, 0, 0), 5, 8, 0)

    if returnImage:
        return image
    else:
        return faces

def pil2cvGrey(pil_im):
    # Convert a PIL image to a greyscale cv image
    # from: http://pythonpath.wordpress.com/2012/05/08/pil-to-opencv-image/
    pil_im = pil_im.convert('L')
    cv_im = cv.CreateImageHeader(pil_im.size, cv.IPL_DEPTH_8U, 1)
    cv.SetData(cv_im, pil_im.tobytes(), pil_im.size[0]  )
    return cv_im

def cv2pil(cv_im):
    # Convert the cv image to a PIL image
    return Image.fromstring("L", cv.GetSize(cv_im), cv_im.tostring())

def imgCrop(image, cropBox, boxScale=1):
    # Crop a PIL image with the provided box [x(left), y(upper), w(width), h(height)]

    # Calculate scale factors
    xDelta=max(cropBox[2]*(boxScale-1),0)
    yDelta=max(cropBox[3]*(boxScale-1),0)

    # Convert cv box to PIL box [left, upper, right, lower]
    PIL_box=[cropBox[0]-xDelta, cropBox[1]-yDelta, cropBox[0]+cropBox[2]+xDelta, cropBox[1]+cropBox[3]+yDelta]

    return image.crop(PIL_box)

def faceCrop(imagePattern,boxScale=1):
    # Select one of the haarcascade files:
    #   haarcascade_frontalface_alt.xml  <-- Best one?
    #   haarcascade_frontalface_alt2.xml
    #   haarcascade_frontalface_alt_tree.xml
    #   haarcascade_frontalface_default.xml
    #   haarcascade_profileface.xml
    faceCascade = cv.Load('haarcascade_frontalface_alt.xml')#cascadeclassifier

    imgList=glob.glob(imagePattern)
    if len(imgList)<=0:
        print ('No Images Found')
        return

    for img in imgList:
        pil_im=Image.open(img)
        cv_im=pil2cvGrey(pil_im)
        faces=DetectFace(cv_im,faceCascade)
        if faces:
            n=1
            for face in faces:
                croppedImage=imgCrop(pil_im, face[0],boxScale=boxScale)
                fname,ext=os.path.splitext(img)
                croppedImage.save(fname+'_crop'+str(n)+ext)
                n+=1
        else:
            print ('No faces found:', img)

def test(imageFilePath):
    pil_im=Image.open(imageFilePath)
    cv_im=pil2cvGrey(pil_im)
    # Select one of the haarcascade files:
    #   haarcascade_frontalface_alt.xml  <-- Best one?
    #   haarcascade_frontalface_alt2.xml
    #   haarcascade_frontalface_alt_tree.xml
    #   haarcascade_frontalface_default.xml
    #   haarcascade_profileface.xml
    faceCascade = cv.Load('haarcascade_frontalface_alt.xml')
    face_im=DetectFace(cv_im,faceCascade, returnImage=True)
    img=cv2pil(face_im)
    img.show()
    img.save('test.png')


# Test the algorithm on an image
#test('testPics/faces.jpg')

# Crop all jpegs in a folder. Note: the code uses glob which follows unix shell rules.
# Use the boxScale to scale the cropping area. 1=opencv box, 2=2x the width and height
def main():
    faceCrop('testPics/*.png', boxScale=1)
    
if __name__ == '__main__':
    main()'''