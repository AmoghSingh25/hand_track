import mediapipe as mp
import numpy as np 
import cv2
from google_text_to_speech import texttospeech
import tensorflow as tf
import keras
from glob import glob


cap=cv2.VideoCapture(0)

mpHands=mp.solutions.hands
hands=mpHands.Hands(max_num_hands=1)
mpDraw=mp.solutions.drawing_utils




def avg_bg(img,bg):
    if bg is None:
        bg = img.copy().astype("float")
        return bg
    
    cv2.accumulateWeighted(img, bg, 0.1)
    return bg



def segment(image,bg_fed, threshold=25):
    image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(bg_fed.astype("uint8"), image)

    thresholded = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)[1]
    thresholded1=thresholded

    (cnts,_) = cv2.findContours(thresholded1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h,w=thresholded.shape
    blank_image=np.zeros((h,w,3), np.uint8)
    if len(cnts) == 0:
        return
    else:
        segmented = max(cnts, key=cv2.contourArea)
        if cv2.contourArea(segmented)>=h*w*0.75:
            return None
        blank_image=cv2.drawContours(blank_image,[segmented],0,(255,255,255),cv2.FILLED)
        blank_image=cv2.cvtColor(blank_image,cv2.COLOR_BGR2GRAY)
        return blank_image


def get_thresh_skin(frame):
    frame = cv2.bilateralFilter(frame, 5, 50, 50)
    
    bgModel = cv2.createBackgroundSubtractorMOG2(0, 50)
    fgmask = bgModel.apply(frame)
    kernel = np.ones((3, 3), np.uint8)
    fgmask = cv2.erode(fgmask, kernel, iterations=1)
    img = cv2.bitwise_and(frame, frame, mask=fgmask)
    
    
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower = np.array([0, 48, 80], dtype="uint8")
    upper = np.array([20, 255, 255], dtype="uint8")
    skinMask = cv2.inRange(hsv, lower, upper)
    
    skinMask1 =skinMask
    contours, hierarchy = cv2.findContours(skinMask1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    length = len(contours)
    maxArea = -1
    if length > 0:
        res = max(contours, key=lambda x: cv2.contourArea(x))
        for i in range(length):
            temp = contours[i]
            area = cv2.contourArea(temp)
            if area > maxArea:
                maxArea = area
                ci = i
                res = contours[ci]
        hull = cv2.convexHull(res)
        drawing = np.zeros(img.shape, np.uint8)
        cv2.drawContours(drawing, [res], 0, (255, 255, 255),cv2.FILLED)
        
        return drawing

def get_skin(img):
    h,w,c=img.shape
    blur = cv2.GaussianBlur(img,(7,7),0)
    blur = cv2.medianBlur(blur, 15)
    img=blur
    hsvim = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower = np.array([0, 48, 80], dtype="uint8")
    upper = np.array([25, 255, 255], dtype="uint8")
    skinRegionHSV = cv2.inRange(hsvim, lower, upper)
    blurred = cv2.blur(skinRegionHSV, (2,2))
    ret,thresh = cv2.threshold(blurred,0,255,cv2.THRESH_BINARY)
    
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    blank_image = np.zeros((h,w,3), np.uint8)
    contours = max(contours, key=lambda x: cv2.contourArea(x))
    cv2.drawContours(blank_image, [contours], -1, (255,255,255), cv2.FILLED)
    cv2.fillPoly(img, pts =contours, color=(255,255,255))
    
    return blank_image

def predict_init(path,model_pt):
    cp_pt="training_"+model_pt+"/cp.ckpt"
    model_pt="saved_model/"+model_pt+"_model"
    new_model = tf.keras.models.load_model(model_pt)
    new_model.summary()
    #new_model.load_weights(cp_pt)
    class_names=1
    return new_model


def start_here():
    let_cl_names=['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
    num_cl_names=['1', '2', '3', '4', '5', '6', '7', '8', '9']
    
    num_model=predict_init("Numbers","numbers")
    let_model=predict_init("Letters","Letters")

    message_fin=""
    msg=None
    msg_p="a"
    n=0
    
    bg=None
    en=True
    while en==True:
        result, img = cap.read()
        img=cv2.flip(img,flipCode=1)
        h,w,c=img.shape
        
        imgRGB=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        results=hands.process(imgRGB)
        blank_image = np.zeros((h,w,3), np.uint8)
        if n<=30:
            if n==0:
                print("Getting background")
            n+=1 
            img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            bg=avg_bg(img,bg)
            if n==30:
                print("Background acquisition completed")
        elif results.multi_hand_landmarks:
            print("Hand located")
            hand_landmarks=results.multi_hand_landmarks[0]

            end_pt=(int(hand_landmarks.landmark[mpHands.HandLandmark.WRIST].x * w+100),int(hand_landmarks.landmark[mpHands.HandLandmark.WRIST].y * h+60))
            start_pt=[int(end_pt[0]-250),int(end_pt[1])-300]

            if(start_pt[1]+20>hand_landmarks.landmark[mpHands.HandLandmark.INDEX_FINGER_TIP].y * h):
                start_pt[1]=int(hand_landmarks.landmark[mpHands.HandLandmark.INDEX_FINGER_TIP].y * h-50)
            
            
            smig=img[start_pt[1]:end_pt[1],start_pt[0]:end_pt[0]]
            bg_fed=bg[start_pt[1]:end_pt[1],start_pt[0]:end_pt[0]]

            if result==True and np.shape(smig)!=0:
                try:
                    timg=get_thresh_skin(smig)
                    #cv2.imshow("Get thresh skin",get_thresh_skin)
                except:
                    timg=0
                temp=get_skin(smig)
                t=segment(smig,bg_fed)
                if t is None:
                    t=temp
                
                cv2.imshow("Back ground",bg_fed.astype("uint8"))
                cv2.imshow("Hand",smig)
                cv2.imshow("segmented",t)
                cv2.imshow("Get skin",temp)
                
                timg=t 
                try:
                    timg=cv2.cvtColor(timg,cv2.COLOR_GRAY2BGR)
                except:
                    n=n
                
                timg = cv2.resize(timg, (50, 50),interpolation = cv2.INTER_AREA)
                temp=cv2.resize(temp, (50, 50),interpolation = cv2.INTER_AREA)
                
                img_array1 = keras.preprocessing.image.img_to_array(timg)
                img_array1 = tf.expand_dims(img_array1, 0) 
                img_array2 = keras.preprocessing.image.img_to_array(temp)
                img_array2 = tf.expand_dims(img_array2, 0) 
                
                prediction_let1 = let_model.predict(img_array1)
                prediction_num1=num_model.predict(img_array1)
                score_let1 = np.max(tf.nn.softmax(prediction_let1[0]))
                score_num1 = np.max(tf.nn.softmax(prediction_num1[0]))
                prediction_let2 = let_model.predict(img_array2)
                prediction_num2=num_model.predict(img_array2)
                score_let2 = np.max(tf.nn.softmax(prediction_let2[0]))
                score_num2 = np.max(tf.nn.softmax(prediction_num2[0]))
                if score_let1>score_let2:
                    conf_let=np.max(score_let1)
                    score_let=score_let1
                else:
                    conf_let=np.max(score_let2)
                    score_let=score_let2
                
                if score_num1>score_num2:
                    conf_num=np.max(score_num1)
                    score_num=score_num1
                else:
                    conf_num=np.max(score_num2)
                    score_num=score_num2
                
                if(conf_let>conf_num):
                    cl=num_cl_names[np.argmax(score_num)]
                    msg="Number="+cl
                else:
                    cl=let_cl_names[np.argmax(score_let)]
                    msg="Letter="+cl
                if(msg!=msg_p):
                    print(msg)
                    msg_p=msg
                blank_image=np.zeros((h,w,3), np.uint8)
                cv2.putText(blank_image,msg,(10,70),cv2.FONT_HERSHEY_PLAIN,3,(255,0,255),3)
                cv2.imshow("Output",blank_image)
                pressed=cv2.waitKey(1)
                if pressed==ord('q'):
                    en=False
                    break
                elif pressed==ord('a'):
                    message_fin+=cl
                    print(message_fin)
                
            else:
                print('out of range')
            
    #texttospeech(message_fin)

start_here()