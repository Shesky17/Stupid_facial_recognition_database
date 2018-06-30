import cv2
import numpy

faceCascade = cv2.CascadeClassifier('/home/pi/opencv-3.4.0/data/haarcascades/haarcascade_frontalface_default.xml')
eyesCascade = cv2.CascadeClassifier('/home/pi/opencv-3.4.0/data/haarcascades/haarcascade_eye.xml')
cap=cv2.VideoCapture(0)
cap.set(3,320)
cap.set(4,240)

while 1:
    ret, img = cap.read()
    img = cv2.flip(img, -1)
    img = cv2.flip(img, 0)
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray, scaleFactor = 1.1, minNeighbors = 15)
    
    for (x,y,w,h) in faces:
        cv2.rectangle(img, (x,y), (x+w, y+h), (0,0,255), thickness = 2)
        roi_gray = gray[x:x+w, y:y+h]
        roi_color = img[x:x+w, y:y+h]
        
        eyes = eyesCascade.detectMultiScale(roi_gray, scaleFactor = 1.1, minNeighbors = 15)

        for(x2,y2,w2,h2) in eyes:
            cv2.rectangle(roi_color, (x2,y2), (x2+w2, y2+h2), (255,0,0), thickness = 2)
        
    #show the image
    cv2.imshow('video windows', img)
    #detect key press
    k = cv2.waitKey(10) & 0xff
    if k == ord('q') or k == 27:
        break
    
#cv2.release()
cam.release()
cv2.destroyAllWindows()