import cv2
import numpy as np
import time
import HandDetectorModule as htm
import autopy

wCam, hCam = 640, 480
pTime = 0
frameR = 100
smooth = 6
wScr, hScr = autopy.screen.size()
plocX, plocY = 0, 0
clocX, clocY = 0, 0

cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)
detect = htm.HandDetector()

while True:
    success, img = cap.read()
    img = detect.findHands(img)
    lmList, bbox = detect.findPositions(img, draw=True)
    
    if len(lmList) > 0:
        x1, y1 = lmList[8][1:]
        x2, y2 = lmList[4][1:]
        
        fingers = detect.fingers()
        
        cv2.rectangle(img, (frameR, frameR), (wCam-frameR, hCam-frameR), (0, 0, 255), 2)
        
        if fingers[1] == 1 and fingers[0] == 1:
            x3 = np.interp(x1, (frameR, wCam-frameR), (0, wScr-20))
            y3 = np.interp(y1, (frameR, hCam-frameR), (0, hScr-20))
            
            clocX = plocX + (x3-plocX)/smooth
            clocY = plocY + (y3-plocY)/smooth
            
            autopy.mouse.move(wScr - clocX, clocY)
            cv2.circle(img, (x1, y1), 5, (255, 255, 0), cv2.FILLED)
            plocX, plocY = clocX, clocY
            
        if fingers[1] == 1 and fingers[0] == 0:
            autopy.mouse.click()
        
    cTime = time.time()
    fps = 1/(cTime - pTime)
    pTime = cTime
    
    cv2.putText(img, f'{str(int(fps))}', (10, 70), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 18), 3)
    cv2.imshow("Image", img)
    cv2.waitKey(1)
