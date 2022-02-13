import cv2 as cv
import numpy as np
import time
import os
import HandTrackingModule as htm

#########################
brushThickness=15
eraserThickness=100
#########################
folderPath="Header"
myList=os.listdir(folderPath)
print(myList)
overlayList=[]
for imPath in myList:
    image=cv.imread(f'{folderPath}/{imPath}')
    overlayList.append(image)
print(len(overlayList))
header=overlayList[0]
drawColor=(255,0,255)

cap=cv.VideoCapture(0)
cap.set(3,1280)
cap.set(4,720)
detector=htm.handDetector(min_detection_confidence=0.85)
xp,yp=0,0
imgCanvas=np.zeros((720,1280,3),np.uint8)
while True:
    #1. import image
    success,img=cap.read()
    img=cv.flip(img,1)
    #2. find hand landmark
    img=detector.findHands(img)
    lmList=detector.findPosition(img,draw=False)
    if len(lmList)!=0:
        #tip of index and middle finger
        x1,y1=lmList[8][1:]
        x2,y2=lmList[12][1:]
        #3. Check which fingers are up
        fingers=detector.fingersUp()
        #4. if selection mode-Two fingers are up
        if fingers[1] and fingers[2]:
            xp,yp=0,0
            print("selection Mode")
            #Checking for the click
            if y1<125:
                if 250<x1<450:
                    header=overlayList[0]
                    drawColor=(255,0,255)
                elif 550<x1<750:
                    header=overlayList[1]
                    drawColor=(255,0,0)
                elif 800<x1<950:
                    header=overlayList[2]
                    drawColor=(0,255,0)
                elif 1050<x1<1200:
                    header=overlayList[3]
                    drawColor=(0,0,0)
            cv.rectangle(img,(x1,y1-25),(x2,y2+25),drawColor,cv.FILLED)
        #5. if drawing mode-Index finger is up
        if fingers[1] and fingers[2]==False:
            cv.circle(img,(x1,y1),15,drawColor,cv.FILLED)
            print("drawing mode")
            if xp==0 and yp==0:
                xp,yp=x1,y1
            if drawColor==(0,0,0):
                cv.line(img,(xp,yp),(x1,y1),drawColor,eraserThickness)
                cv.line(imgCanvas,(xp,yp),(x1,y1),drawColor,eraserThickness)
            else:
                cv.line(img,(xp,yp),(x1,y1),drawColor,brushThickness)
                cv.line(imgCanvas,(xp,yp),(x1,y1),drawColor,brushThickness)
            xp,yp=x1,y1

    imgGrey=cv.cvtColor(imgCanvas,cv.COLOR_BGR2GRAY)
    _,imgInv=cv.threshold(imgGrey,50,255,cv.THRESH_BINARY_INV)
    imgInv=cv.cvtColor(imgInv,cv.COLOR_GRAY2BGR)
    img=cv.bitwise_and(img,imgInv)
    img=cv.bitwise_or(img,imgCanvas)
    #setting the header image
    img[0:125,0:1280]=header
    #img=cv.addWeighted(img,0.5,imgCanvas,0.5,0)
    cv.imshow("Image",img)
    cv.imshow("Canvas",imgCanvas)
    cv.imshow("Inv",imgInv)
    cv.waitKey(1)