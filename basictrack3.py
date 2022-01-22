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
#from google_text_to_speech import texttospeech
import ctypes
import tensorflow as tf

start_pt=None
end_pt=None
bg = None
mpHands=mp.solutions.hands
hands=mpHands.Hands(max_num_hands=1)
mpDraw=mp.solutions.drawing_utils

def predict_init(ch):
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
    # new_model = tf.saved_model.load('saved_model/work_model_1')
    # new_model.summary()
    new_model.load_weights(checkpoint_path)
    loss, acc = new_model.evaluate(val_ds, verbose=2)
    if ch==0:
        return new_model
    elif ch==1:
        return class_names


def run_avg(image, aWeight):
    global bg
    if bg is None:
        bg = image.copy().astype("float")
        return

    cv2.accumulateWeighted(image, bg, aWeight)

def get_mediapipe(img, pt=False):
    imgRGB=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    h,w,c=img.shape
    results=hands.process(imgRGB)
    """ timg = np.zeros((h,w,3), np.uint8) """
    smig=None
    del imgRGB
    if results.multi_hand_landmarks:
        """ for hand_landmarks in results.multi_hand_landmarks:
            mpDraw.draw_landmarks(
            timg, hand_landmarks, mpHands.HAND_CONNECTIONS) """
        hand_landmarks=results.multi_hand_landmarks[0]

        x_c=hand_landmarks.landmark[mpHands.HandLandmark.WRIST].x*w +hand_landmarks.landmark[mpHands.HandLandmark.MIDDLE_FINGER_TIP].x*w
        y_c=hand_landmarks.landmark[mpHands.HandLandmark.WRIST].y*h +hand_landmarks.landmark[mpHands.HandLandmark.MIDDLE_FINGER_TIP].y*h
        
        mid_pt=(x_c/2,y_c/2)

        global start_pt
        start_pt=(int(mid_pt[0]-w/4),int(mid_pt[1]-h/4))
        global end_pt
        end_pt=(int(mid_pt[0]+w/4),int(mid_pt[1]+h/4))


        """ end_pt=(int(hand_landmarks.landmark[mpHands.HandLandmark.WRIST].x * w+100),int(hand_landmarks.landmark[mpHands.HandLandmark.WRIST].y * h+60))
        start_pt=tuple([int(end_pt[0]-250),int(end_pt[1])-300]) """
        """ if(start_pt[1]+20>hand_landmarks.landmark[mpHands.HandLandmark.INDEX_FINGER_TIP].y * h):
            start_pt[1]=int(hand_landmarks.landmark[mpHands.HandLandmark.INDEX_FINGER_TIP].y * h-50)
         """
        smig=img[start_pt[1]:abs(end_pt[1]),start_pt[0]:end_pt[0]]
        if np.shape(smig) != () and smig is not None:
            #print(start_pt,end_pt)
            smig=cv2.cvtColor(smig,cv2.COLOR_BGR2GRAY)
            smig=cv2.resize(smig,(50,50),interpolation=cv2.INTER_AREA)
    return smig


def segment(image, threshold=16):
    global bg
    # find the absolute difference between background and current frame
    diff = cv2.absdiff(bg.astype("uint8"), image)

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


if __name__ == "__main__":
    class_names=predict_init(1)
    new_model=predict_init(0)
    
    aWeight = 0.5

    
    camera = cv2.VideoCapture(0)

    top, right, bottom, left = 10, 350, 225, 590

    num_frames = 0

    while(True):
        (grabbed, frame) = camera.read()

        frame = imutils.resize(frame, width=700)

        frame = cv2.flip(frame, 1)

        gray=None

        clone = frame.copy()
        (height, width) = frame.shape[:2]
        # convert the roi to grayscale and blur it
        img_pass=get_mediapipe(frame)
        if np.shape(img_pass) != ():
            #start_pt,end_pt=get_mediapipe(frame,True)
            left,top=start_pt[0],start_pt[1]
            right,bottom=end_pt[0],start_pt[1]

        roi = frame[top:bottom, right:left]
       
        if np.shape(img_pass) != ():
            """ gray = cv2.cvtColor(img_pass, cv2.COLOR_BGR2GRAY) """
            gray = cv2.GaussianBlur(img_pass, (7, 7), 0)
        
        # to get the background, keep looking till a threshold is reached
        # so that our running average model gets calibrated
        if num_frames < 30 and gray is not None:
            run_avg(gray, aWeight)
        else:
            """ img_pass=get_mediapipe(frame) """
            # segment the hand region
            hand=None
            if img_pass is not None and bg is not None:
                hand = segment(img_pass)

            # check whether hand region is segmented
            if hand is not None:
                # if yes, unpack the thresholded image and
                # segmented region
                (thresholded, segmented) = hand

                # draw the segmented region and display the frame
                cv2.drawContours(clone, [segmented + (right, top)], -1, (0, 0, 255))
                cv2.imshow("Thesholded", thresholded)
                thresholded=cv2.cvtColor(thresholded,cv2.COLOR_GRAY2BGR)
                #thresholded=cv2.resize(thresholded,(50,50), interpolation=cv2.INTER_AREA)
                img_array = keras.preprocessing.image.img_to_array(thresholded)
                img_array = tf.expand_dims(img_array, 0)
            
                predictions = new_model.predict(img_array)
                score = tf.nn.softmax(predictions[0])
                #texttospeech(str("Class - "+str(class_names[np.argmax(score)])))
                # print(class_names[np.argmax(score)])


        # draw the segmented hand
        try:
            cv2.rectangle(clone, tuple(start_pt),tuple(end_pt), (0,255,0), 2)
        except:
            continue

        # increment the number of frames
        num_frames += 1

        # display the frame with segmented hand
        cv2.imshow("Video Feed", clone)

        # observe the keypress by the user
        keypress = cv2.waitKey(1) & 0xFF

        # if the user pressed "q", then stop looping
        if keypress == ord("q"):
            break

# free up memory
camera.release()
cv2.destroyAllWindows()