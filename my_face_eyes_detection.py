import numpy as np
import cv2

faceCascade = cv2.CascadeClassifier('Cascades/haarcascade_frontalface_default.xml')
eyeCascade = cv2.CascadeClassifier('Cascades/haarcascade_eye.xml')
faceCascade.load('/home/opencv-3.4.0/data/haarcascades/haarcascade_frontalface_default.xml')
eyeCascade.load('/home/opencv-3.4.0/data/haarcascades/haarcascade_eye.xml')
cap = cv2.VideoCapture(0)
cap.set(3,640) # 3 for width
cap.set(4,480) # 4 for height

while(True):
    ret, colorImg = cap.read()
    colorImg = cv2.flip(colorImg, 0)
    colorImg = cv2.flip(colorImg, -1)
    grayImg = cv2.cvtColor(colorImg, cv2.COLOR_BGR2GRAY)
    
    faces = faceCascade.detectMultiScale(
        grayImg,
        scaleFactor = 1.2,
        minNeighbors = 5
        )
    
    for (x,y,w,h) in faces:
        cv2.rectangle(colorImg, (x,y), (x+w, y+h), (255,0,0), 2)
        roi_gray = grayImg[y:y+h, x:x+w]
        roi_color = colorImg[y:y+h, x:x+w]
        
        eyes = eyeCascade.detectMultiScale(roi_gray)
        for (x2,y2,w2,h2) in eyes:
            cv2.rectangle(roi_color, (x2,y2), (x2 + w2, y2 + h2), (0,255,0), 2)
        
    cv2.imshow('video', colorImg)
    #cv2.imshow('video', grayImg)
    
    # press 'ESC' to quit
    k = cv2.waitKey(30) & 0xff
    if k==27:
        break
    
cap.release()
cv2.destroyAllWindows()