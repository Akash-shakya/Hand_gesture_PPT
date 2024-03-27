#from cvzone.HandTrackingModule import HandDetector
import cv2
import os
#import numpy as np

# Parameters
width, height = 1280, 720
#gestureThreshold = 300
folderPath = "D:\\Python_projects\\Hand_gesture_PPT\\ppt"

# Camera Setup
cap = cv2.VideoCapture(0)
cap.set(3, width)
cap.set(4, height)

# Get list of presentation images
pathImages = sorted(os.listdir(folderPath), key=len)
print(pathImages)

while True:
    # Get image frame
    success, img = cap.read()
    cv2.imshow("Image",img)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

