import cv2
import mediapipe as mp
import time
import numpy as np 
import matplotlib.pyplot as plt
import numpy as np
import os
import imutils
import PIL
import keras
import cv2
import time
import pathlib
from google_text_to_speech import texttospeech
import ctypes
import tensorflow as tf



cap=cv2.VideoCapture(0)
mpHands=mp.solutions.hands
hands=mpHands.Hands(max_num_hands=1)
mpDraw=mp.solutions.drawing_utils
bg = None
pTime=0
cTime=0


def bag(img):
    foreground = img
    r=img
    foreground = cv2.cvtColor(foreground, cv2.COLOR_BGR2RGB)
    foreground = cv2.resize(foreground,(r.shape[1],r.shape[0]))
    background = 255 * np.ones_like(r).astype(np.uint8)
    foreground = foreground.astype(float)
    background = background.astype(float)
    th, alpha = cv2.threshold(np.array(r),0,255, cv2.THRESH_BINARY)
    alpha = cv2.GaussianBlur(alpha, (7,7),0)
    alpha = alpha.astype(float)/255
    foreground = cv2.multiply(alpha, foreground)
    background = cv2.multiply(1.0 - alpha, background)
    outImage = cv2.add(foreground, background)
    return outImage/255

""" def get_img_contour_thresh(img):
	img = cv2.flip(img, 1)
	imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
	dst = cv2.calcBackProject([imgHSV], [0, 1], hist, [0, 180, 0, 256], 1)
	disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(10,10))
	cv2.filter2D(dst,-1,disc,dst)
	blur = cv2.GaussianBlur(dst, (11,11), 0)
	blur = cv2.medianBlur(blur, 15)
	thresh = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
	thresh = cv2.merge((thresh,thresh,thresh))
	thresh = cv2.cvtColor(thresh, cv2.COLOR_BGR2GRAY)
	thresh = thresh[y:y+h, x:x+w]
	contours = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[0]
	return img, contours, thresh
 """

def run_avg(image, aWeight):
    global bg
    # initialize the background
    if bg is None:
        bg = image.copy().astype("float")
        return

    # compute weighted average, accumulate it and update the background
    cv2.accumulateWeighted(image, bg, aWeight)


def predict_init():
    h=w=50
    data_dir=pathlib.Path("img")
    image_count = len(list(data_dir.glob('*/*.jpg')))
    batch_size = 32
    img_height = h
    img_width = w
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size)
    AUTOTUNE = tf.data.AUTOTUNE
    checkpoint_path = "training_1/cp.ckpt"
    class_names = train_ds.class_names
    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size)

    new_model = tf.keras.models.load_model('saved_model/model2')
    new_model.summary()
    new_model.load_weights(checkpoint_path)
    loss, acc = new_model.evaluate(val_ds, verbose=2)
    return new_model,class_names


def get_hand(img):
    h,w,c=img.shape
    blur = cv2.GaussianBlur(img,(7,7),1)
    blur = cv2.medianBlur(blur, 15)
    img=blur
    hsvim = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower = np.array([0, 40, 70], dtype = "uint8")
    upper = np.array([50, 255, 255], dtype = "uint8")
    skinRegionHSV = cv2.inRange(hsvim, lower, upper)
    blurred = cv2.blur(skinRegionHSV, (2,2))
    ret,thresh = cv2.threshold(blurred,0,255,cv2.THRESH_BINARY)
    #return thresh
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    blank_image = np.zeros((h,w,3), np.uint8)
    contours = max(contours, key=lambda x: cv2.contourArea(x))
    cv2.drawContours(blank_image, [contours], -1, (255,255,255), cv2.FILLED)
    cv2.fillPoly(img, pts =contours, color=(255,255,255))
    #cv2.imshow("contours", img)
    return blank_image




def filter_img(img):
    img=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    (thresh, blackAndWhiteImage) = cv2.threshold(img, 130, 255, cv2.THRESH_BINARY)
    #blackAndWhiteImage = cv2.bitwise_not(blackAndWhiteImage)
    return blackAndWhiteImage

def segment(image, threshold=16):
    global bg
    # find the absolute difference between background and current frame
    diff = cv2.absdiff(bg.astype("uint8"), image)

    #return diff

    # threshold the diff image so that we get the foreground
    thresholded = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)[1]
    # get the contours in the thresholded image
    (cnts,_) = cv2.findContours(thresholded.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # return None, if no contours detected
    if len(cnts) == 0:
        return
    else:
        # based on contour area, get the maximum contour which is the hand
        segmented = max(cnts, key=cv2.contourArea)
        return (thresholded, segmented)


def cont(img):
    h,w,c=img.shape
    img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,1)
    #ret,thresh=cv2.threshold(img,100,255,cv2.THRESH_BINARY_INV)
    #return thresh
    blank_image = np.zeros((h,w,3), np.uint8)
    contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    #cv2.drawContours(img, contours, -1, (255,255,255), 3)
    if(len(contours)!=0):
        cn=max(contours,key=cv2.contourArea)
        #cv2.drawContours(img, cn, -1, (255,255,255),1)
        cv2.fillPoly(blank_image, pts =[cn], color=(255,255,255))
    img=filter_img(blank_image)
    img=cv2.flip(img,flipCode=1)
    return img

ret_init=predict_init()
new_model,class_names=ret_init[0],ret_init[1]
num_frames=0
result, timg = cap.read()
while True:
    result, timg = cap.read()
    img=timg
    img=cv2.flip(img,flipCode=1)
    h,w,c=img.shape
    flag=1
    imgRGB=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    results=hands.process(imgRGB)
    blank_image = np.zeros((h,w,3), np.uint8)
    aWeight=0.5
    gray = cv2.cvtColor(timg, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)
    if num_frames < 30:
        run_avg(gray, aWeight)
    else:
         # segment the hand region
        hand = segment(gray)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mpDraw.draw_landmarks(
            timg, hand_landmarks, mpHands.HAND_CONNECTIONS)
        hand_landmarks=results.multi_hand_landmarks[0]
        end_pt=(int(hand_landmarks.landmark[mpHands.HandLandmark.WRIST].x * w+100),int(hand_landmarks.landmark[mpHands.HandLandmark.WRIST].y * h+60))
        start_pt=[int(end_pt[0]-250),int(end_pt[1])-300]
        if(start_pt[1]+20>hand_landmarks.landmark[mpHands.HandLandmark.INDEX_FINGER_TIP].y * h):
            start_pt[1]=int(hand_landmarks.landmark[mpHands.HandLandmark.INDEX_FINGER_TIP].y * h-50)
        smig=img[start_pt[1]:end_pt[1],start_pt[0]:end_pt[0]]
        if True:
            temp_img=get_hand(smig)

            

            #temp_img=cv2.cvtColor(temp_img,cv2.COLOR_BGR2GRAY)
            img_array = keras.preprocessing.image.img_to_array(temp_img)
            img_array = tf.expand_dims(img_array, 0)
            
            predictions = new_model.predict(img_array)
            score = tf.nn.softmax(predictions[0])
            #texttospeech(str("Class - "+str(class_names[np.argmax(score)])))
            print(class_names[np.argmax(score)])
            cv2.waitKey(1)
        else:
            if(flag==1):
                flag=0
                print("Not detected") 
            

    