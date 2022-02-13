import cv2 as cv
import mediapipe as mp
import time



class handDetector():
    def __init__(self,static_image_mode=False,max_num_hands=2,model_complexity=1,min_detection_confidence: float = 0.5,min_tracking_confidence: float = 0.5):
        self.mode=static_image_mode
        self.maxHands=max_num_hands
        self.modelCom=model_complexity
        self.trackCon=min_tracking_confidence
        self.detCon=min_detection_confidence
        self.mpHands=mp.solutions.hands
        self.hands=self.mpHands.Hands(self.mode,self.maxHands,self.modelCom,self.detCon,self.trackCon)
        self.mpDraw=mp.solutions.drawing_utils
        self.tipIds=[4,8,12,16,20]
    def findHands(self,img,draw=True):
        imgRGB=cv.cvtColor(img,cv.COLOR_BGR2RGB)
        self.results=self.hands.process(imgRGB)
        #print(results.multi_hand_landmarks)
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img,handLms,self.mpHands.HAND_CONNECTIONS)
        return img

    def findPosition(self,img,handNo=0,draw=True):
        self.lmList=[]
        if self.results.multi_hand_landmarks:
            myHand=self.results.multi_hand_landmarks[handNo]
            for id,lm in enumerate(myHand.landmark):
                h,w,c=img.shape
                cx,cy=int(lm.x*w),int(lm.y*h)
                #print(id,cx,cy)
                self.lmList.append([id,cx,cy])
                if draw:
                    cv.circle(img,(cx,cy),10,(255,0,255),cv.FILLED)
        return self.lmList

    def fingersUp(self):
        fingers=[]
        #Thumbs
        if self.lmList[self.tipIds[0]][1]<self.lmList[self.tipIds[0]-1][1]:
            fingers.append(1)
        else:
            fingers.append(0)
        #fingers
        for id in range(1,5):
            if self.lmList[self.tipIds[id]][2]<self.lmList[self.tipIds[id]-2][2]:
                fingers.append(1)
            else:
                fingers.append(0)

        return fingers


def main():
    pTime=0
    cap=cv.VideoCapture(0)
    detector=handDetector()
    while True:
        success,img=cap.read()
        img=detector.findHands(img)
        lmList=detector.findPosition(img)
        if len(lmList)!=0:
            print(lmList[4])

        cTime=time.time()
        fps=1/(cTime-pTime)
        pTime=cTime

        cv.putText(img,str(int(fps)),(10,70),cv.FONT_HERSHEY_PLAIN,3,(255,0,255),3)
        cv.imshow("image",img)
        cv.waitKey(1)


if __name__ == "__main__":
    main()