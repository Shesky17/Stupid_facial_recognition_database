import cv2
import os

cam = cv2.VideoCapture(0)
cam.set(3,320)
cam.set(4,240)
face_detector = cv2.CascadeClassifier('/home/opencv-3.4.0/data/haarcascades/haarcascade_frontalface_default.xml')

usr_id = input('\n enter usr id (numbers only) and press <Enter> ==> ') # input of name
usr_folder_name = str(usr_id) 
str_path = '/home/ProjectFace/database/' + usr_folder_name # it's a string
path_path = os.path.join('/home/ProjectFace/database/', usr_folder_name) # it's a path object

# if usr entered empyt name, close the program
try:
    input = usr_id
    if not input:
        raise ValueError('\n\tEmpty string' +
                         '\n\tPlease enter a valid usr ID' +
                         '\n\tExiting the program.....\n')
except ValueError as e:
    print(e)
    cam.release()
    cv2.destroyAllWindows()

# Say not more counting dollars
# We'll be counting faces
count = 0 
print("\n[ INFO ] Initializing the program. Please stare at the camera and wait...")

while (True):
    ret, img = cam.read()
    img =cv2.flip(img, -1)
    img = cv2.flip(img, 0)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, scaleFactor = 1.2, minNeighbors = 10)
    
    for (x,y,w,h) in faces:
        #print('its the for loop: ')
        cv2.rectangle(img, (x,y), (x+w, y+h), (255,0,0), 2)
        count += 1;
        roi_gray = gray[y:y+h, x:x+h]
        roi_color = img[y:y+h, x:x+h]
        str_final_name = str_path + '_' + str(count) + '.jpg'
        
        # save the captured image into the database folder
        if os.path.exists(str_path):
            print("\n\tFolder exists. Saving picture " + str(count) + " into the folder... \n")
            os.chdir(str_path)
            cv2.imwrite(str(usr_id) + '_' + str(count) + '.jpg', roi_gray)
        if not os.path.exists(str_path):
            print("\n\tFirst time User. Creating new folder.. \n")
            os.mkdir(str_path)
            os.chdir(str_path)
            cv2.imwrite(str(usr_id) + '_' + str(count) + '.jpg', roi_gray)
        cv2.imshow('image', img)
        
    # program will be closed if press 'Esc' key
    k = cv2.waitKey(100) & 0xff 
    if k == 27: 
        break
    # program will be xlosed after count = 30
    elif count>=30: 
        break
    
print('\n\tFace captured! \n\tExiting the program!\n\n')
cam.release()
cv2.destroyAllWindows()