import cv2
import mediapipe as mp
import time
import numpy as np 
import matplotlib.pyplot as plt
import numpy as np
import os
import imutils
import PIL
import tensorflow as tf
import cv2
import time
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import pathlib
from google_text_to_speech import texttospeech
#from keras.backend import manual_variable_initialization manual_variable_initialization(True)import imutils 





cap=cv2.VideoCapture(0)

mpHands=mp.solutions.hands
hands=mpHands.Hands(max_num_hands=1)
mpDraw=mp.solutions.drawing_utils


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


def filter_img(img):
    #img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    (thresh, blackAndWhiteImage) = cv2.threshold(img, 130, 255, cv2.THRESH_BINARY)
    return blackAndWhiteImage

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




img=cv2.imread("Image/0/1.jpg")
h,w,c=img.shape
del img
data_dir=pathlib.Path("Image")
image_count = len(list(data_dir.glob('*/*.jpg')))
print(image_count)
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

new_model = tf.keras.models.load_model('saved_model/model3')
new_model.summary()
new_model.load_weights(checkpoint_path)
loss, acc = new_model.evaluate(val_ds, verbose=2)
print("Restored model, accuracy: {:5.2f}%".format(100 * acc))
while True:
    result, img = cap.read()
    img=cv2.flip(img,flipCode=1)
    h,w,c=img.shape
    '''smig=img[10:int(h/2)+10,int(w/2):int(0.9*w)]
    smig=cv2.cvtColor(smig,cv2.COLOR_BGR2GRAY)
    blur=cv2.GaussianBlur(smig,(5,5),2)
    th3 = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,11,2)
    ret, res = cv2.threshold(th3, 80, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    smig=res'''
    imgRGB=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    results=hands.process(imgRGB)
    blank_image = np.zeros((h,w,3), np.uint8)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
           # print(
        #  f'Index finger tip coordinates: (',
        #  f'{hand_landmarks.landmark[mpHands.HandLandmark.INDEX_FINGER_TIP].x * w}, '
        #  f'{hand_landmarks.landmark[mpHands.HandLandmark.INDEX_FINGER_TIP].y * h})'
     # )

            mpDraw.draw_landmarks(
            blank_image, hand_landmarks, mpHands.HAND_CONNECTIONS)
            
            end_pt=(int(hand_landmarks.landmark[mpHands.HandLandmark.WRIST].x * w+100),int(hand_landmarks.landmark[mpHands.HandLandmark.WRIST].y * h+60))
            start_pt=[int(end_pt[0]-250),int(end_pt[1])-300]
            if(start_pt[1]+20>hand_landmarks.landmark[mpHands.HandLandmark.INDEX_FINGER_TIP].y * h):
                start_pt[1]=int(hand_landmarks.landmark[mpHands.HandLandmark.INDEX_FINGER_TIP].y * h-50)
            
            smig=blank_image[start_pt[1]:end_pt[1],start_pt[0]:end_pt[0]]
            #cv2.rectangle(img,tuple(start_pt),end_pt,color=(255,255,255),thickness=3)
            try:
                cv2.imshow("Image", smig)
                temp=
                h,w,c=img.shape
                img_height = h
                img_width = w
                temp = cv2.resize(temp, (50, 50),
                            interpolation = cv2.INTER_AREA)
                img=temp

                img_array = keras.preprocessing.image.img_to_array(img)
                img_array = tf.expand_dims(img_array, 0) # Create a batch

                predictions = new_model.predict(img_array)
                score = tf.nn.softmax(predictions[0])
                #texttospeech(str("Class - "+str(class_names[np.argmax(score)])))
                print(class_names[np.argmax(score)])
                #cv2.imshow("Image whole ", img)
                cv2.waitKey(1)

            except:
                print('out of range')
            
            

    cTime=time.time()
    fps=1/(cTime-pTime)
    pTime=cTime
    cv2.putText(img, str(int(fps)),(10,70), cv2.FONT_HERSHEY_PLAIN,3,(255,0,255),3)
    
    #cv2.imshow("Small image", smig)