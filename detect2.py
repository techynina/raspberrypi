from picamera.array import PiRGBArray
from picamera import PiCamera
import cv2
import time
import os
import sys
import RPi.GPIO as IO            
import time                              

IO.setmode (IO.BOARD)       
IO.setup(40,IO.OUT) 
IO.output(40,0)
time.sleep(1)

camera = PiCamera()
camera.resolution = (320, 240)
camera.framerate = 30
rawCapture = PiRGBArray(camera, size=(320, 240))

display_window = cv2.namedWindow("Human_Detection")

#face classifier
pathtoface = os.path.join(sys.path[0], 'face_classifier.xml')
face_cascade = cv2.CascadeClassifier(pathtoface)

#full body classifier
pathtobody = os.path.join(sys.path[0], 'full_body.xml')
body_cascade = cv2.CascadeClassifier(pathtobody)

#eyebrow classifier


time.sleep(1)

for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):

    image = frame.array
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

    #FACE DETECTION STUFF
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)
    for (x,y,w,h) in faces:
        cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,0),2)
        IO.output(40,1)
        time.sleep(3)
        IO.output(40,0)
    #BODY DETECTION STUFF
    bodies = body_cascade.detectMultiScale(gray, 1.1, 5)
    for (x,y,w,h) in bodies:
        cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,0),2)
        IO.output(40,1)
        time.sleep(3) 
        IO.output(40,0)

    if faces is None or bodies is None:
        IO.output(40,0)        
        time.sleep(1) 

    #DISPLAY TO WINDOW
    cv2.imshow("Human_Detection", image)
    key = cv2.waitKey(1)

    rawCapture.truncate(0)

    if key == ord("q"):
        camera.close()
        cv2.destroyAllWindows()
        break
