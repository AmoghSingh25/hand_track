import pathlib
import mediapipe as mp
import glob
import cv2
import numpy as np 


mpHands=mp.solutions.hands
hands=mpHands.Hands(max_num_hands=1)
mpDraw=mp.solutions.drawing_utils


i=5
while(i<26):
    ch='F'
    ch=ord(ch)+i
    ch=chr(ch)
    data_dir="D:/Projects/Hand_track/newimg/"+ch+"/1.jpg"
    print(data_dir)
    img= cv2.imread(data_dir,cv2.IMREAD_COLOR)
    h,w,c=img.shape
    blank_image = np.zeros((h,w,3), np.uint8)
    
    cv2.waitKey(0)
    i+=1
    imgRGB=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    results=hands.process(imgRGB)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mpDraw.draw_landmarks(
            blank_image, hand_landmarks, mpHands.HAND_CONNECTIONS)
        temp = cv2.resize(blank_image, (50, 50),
                            interpolation = cv2.INTER_AREA)
        filename="D:/Projects/Hand_track/newimg/"+ch+"/2.jpg"
        cv2.imshow("Image1",img)
        cv2.imshow("Image", temp)