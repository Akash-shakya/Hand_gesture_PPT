from cvzone.HandTrackingModule import HandDetector
import cv2
import os
#import numpy as np

# Parameters
width, height = 1280, 720
gestureThreshold = 300
folderPath = "D:\\Python_projects\\Hand_gesture_PPT\\ppt"

# Camera Setup
cap = cv2.VideoCapture(0)
cap.set(3, width)
cap.set(4, height)

# Variables
# imgList = []
delay = 30
buttonPressed = False
counter = 0
# drawMode = False
imgNumber = 0
#hs, ws = int(120 *1), int(213 * 1)  # width and height of small image

# Hand Detector
detectorHand = HandDetector(detectionCon=0.8, maxHands=1)


# Get list of presentation images
pathImages = sorted(os.listdir(folderPath), key=len)
#print(pathImages)

while True:
    # Get image frame
    success, img = cap.read()
    img = cv2.flip(img, 1)
    pathFullImage = os.path.join(folderPath, pathImages[imgNumber])
    imgCurrent = cv2.imread(pathFullImage)

    # Find the hand and its landmarks
    hands, img = detectorHand.findHands(img)  # with draw
    
    # Draw Gesture Threshold line
    cv2.line(img, (0, gestureThreshold), (width, gestureThreshold), (0, 255, 0), 10)

    if hands and buttonPressed is False: # If hand is detected

        hand = hands[0]
        fingers = detectorHand.fingersUp(hand)  # List of which fingers are up/
        cx, cy = hand["center"]
        lmList = hand["lmList"]  # List of 21 Landmark points
        indexFinger = lmList[8][0] , lmList[8][1]

        cx, cy = hand["center"]
        #print(fingers)

        if cy <= gestureThreshold:  # If hand is at the height of the face
            
            #gestures 1 move left page
            if fingers == [1, 0, 0, 0, 0]:
                print("Left")
                if imgNumber > 0:
                    buttonPressed = True
                    imgNumber -= 1

            #gestures 2 move right page         
            if fingers == [0, 0, 0, 0, 1]:
                print("Right")
                if imgNumber < len(pathImages) - 1:
                    buttonPressed = True
                    imgNumber += 1

        #gesture 3 pointer
        if fingers == [0, 1, 1, 0, 0]:
            cv2.circle(imgCurrent, indexFinger, 12, (0, 0, 255), cv2.FILLED)
    if buttonPressed:
        counter += 1
        if counter > delay:
            counter = 0
            buttonPressed = False

    # Define the new dimensions (width, height)
    new_width = 220
    new_height = 120

    # Resize the image
    resized_image = cv2.resize(img, (new_width, new_height))

    # Get the dimensions of the small image
    small_height, small_width, _ = resized_image.shape

    # Define the coordinates to place the small image in the top right corner
    top_right_x = 2 # 10 pixels margin from the right edge
    top_right_y = 2  # 10 pixels margin from the top edge

    # Overlay the small image onto the main image
    imgCurrent[top_right_y:top_right_y+small_height, top_right_x:top_right_x+small_width] = resized_image

    

    cv2.imshow("Image",img)
    cv2.imshow("Slide",imgCurrent)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

