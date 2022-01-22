import cv2
import mediapipe as mp
import time
import numpy as np
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt



def grab(img,rect):
    mask = np.zeros(img.shape[:2],np.uint8)
    bgdModel = np.zeros((1,65),np.float64)
    fgdModel = np.zeros((1,65),np.float64)
    cv.grabCut(img,mask,rect,bgdModel,fgdModel,2,cv.GC_INIT_WITH_RECT)
    mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
    img = img*mask2[:,:,np.newaxis]
    return img

cap=cv2.VideoCapture(0)
results=[]
mpHands=mp.solutions.hands
hands=mpHands.Hands()
mpDraw=mp.solutions.drawing_utils

def find_cov():
    f=open("data.txt",'r')
    orig=[]
    i=0
    for data in f:
        dt=data.split(' ')
        x,y,z=float(dt[1]),float(dt[2]),float(dt[3])
        temp=[x,y,z]
        orig.append(temp)
    f.close()
    f=open("data2.txt",'r')
    nx=[]
    i=0
    for data in f:
        dt=data.split(' ')
        x,y,z=float(dt[1]),float(dt[2]),float(dt[3])
        temp=[x,y,z]
        nx.append(temp)
    f.close()
    for i in range(0,len(orig)):
        x1,y1,z1=orig[i][1],orig[i][2],orig[i][3]
        x2,y2,z2=nx[i][1],nx[i][2],nx[i][3]
        print(np.cov(x1,x2))



def write_data():
    f=open("data2.txt",'a')
    for handLms in results.multi_hand_landmarks:
        for id, lm in enumerate(handLms.landmark):
            x,y,z=str(lm.x),str(lm.y),str(lm.z)
            s=str(id)+" "+x+" "+y+" "+z+"\n"
            f.write(s)
    f.close()
        
            


def ret_coords():
    zero=[]
    for handLms in results.multi_hand_landmarks:
        for id, lm in enumerate(handLms.landmark):
            if(id==0):
                zero=lm
            else:
                x1=lm.x
                y1=lm.y
                z1=lm.z
                x2=zero.x
                y2=zero.y
                z2=zero.z
                x=x1-x2
                y=y1-y2
                z=z1-z2
                lm.x, lm.y , lm.z = x,y,z
    #print(results.multi_hand_landmarks)
                

pTime=0
cTime=0

while True:
    result, img = cap.read()
    h,w,_=img.shape
    imgRGB=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    results=hands.process(imgRGB)
    if results.multi_hand_landmarks:
        hand_landmarks=results.multi_hand_landmarks[0]
        
        x_c=hand_landmarks.landmark[mpHands.HandLandmark.WRIST].x*w +hand_landmarks.landmark[mpHands.HandLandmark.MIDDLE_FINGER_TIP].x*w
        y_c=hand_landmarks.landmark[mpHands.HandLandmark.WRIST].y*h +hand_landmarks.landmark[mpHands.HandLandmark.MIDDLE_FINGER_TIP].y*h
        
        mid_pt=(x_c/2,y_c/2)

        start_pt=(int(mid_pt[0]-w/4),int(mid_pt[1]-h/4))
        end_pt=(int(mid_pt[0]+w/4),int(mid_pt[1]+h/4))

        rect=(start_pt[0],start_pt[1],end_pt[0],end_pt[1])

        cv2.rectangle(img,tuple(start_pt),tuple(end_pt),color=(255,255,255),thickness=3)

        


        

    cv2.imshow("Image",img)
    cv2.waitKey(1)