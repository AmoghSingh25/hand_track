import cv2
import mediapipe as mp
import time
import numpy as np


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

if True:
    img=cv2.imread("A.jpg")
    imgRGB=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    results=hands.process(imgRGB)
    #print(len(results.multi_hand_landmarks))
    #print(results.multi_hand_landmarks)
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            for id, lm in enumerate(handLms.landmark):
                h,w,c=img.shape
                cx,cy=int(lm.x*w), int(lm.y*h)
                if(id==0):
                    cv2.circle(img, (cx,cy), 25, (255,0,255), cv2.FILLED)

            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)
        #print("New coords")
        ret_coords()
        #write_data()
        find_cov()
    cTime=time.time()
    fps=1/(cTime-pTime)
    pTime=cTime
    cv2.putText(img, str(int(fps)),(10,70), cv2.FONT_HERSHEY_PLAIN,3,(255,0,255),3)
    cv2.imshow("Image", img)
    #cv2.waitKey(0)