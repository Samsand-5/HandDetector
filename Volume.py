import cv2
import numpy as np
import time
import HandDetectorModule as htm
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume


wCam, hCam = 640, 480
detector = htm.HandDetector(detectionCon = 0.7)

cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)
pTime = 0
area = 0

devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)

volume = cast(interface, POINTER(IAudioEndpointVolume))
volumeRange = volume.GetVolumeRange()
volume.SetMasterVolumeLevel(0, None)
minVol, maxVol = volumeRange[0], volumeRange[1]
vol, volBar, volPer = 0, 150, 0
colorVol = (255, 0, 0)

while True:
    success, img = cap.read()
    img = detector.findHands(img)
    lmList, bbox = detector.findPositions(img, draw=True)
    
    if len(lmList) != 0:
        
        area = abs(bbox[2] - bbox[0]) * abs(bbox[3] - bbox[1]) // 100
        if 250 < area < 1200:
            
            length, img, lineInfo = detector.findDistance(4, 8, img)
            
            vol = np.interp(length, [60, 250], [minVol, maxVol])
            volBar = np.interp(length, [60, 250], [400, 150])
            volPer = np.interp(length, [60, 250], [0, 100])
            
            smooth = 10
            volPer = smooth * round(volPer/smooth)
            
            fingers = detector.fingers()
            
            if not fingers[4]:
                volume.SetMasterVolumeLevelScalar(volPer/100, None)
                cv2.line(img, (lineInfo[0], lineInfo[1]), (lineInfo[2], lineInfo[3]), (0, 0, 255), 3)
                colorVol = (0, 255, 0)
                
            else:
                colorVol = (255, 0, 0)
            
    cv2.rectangle(img, (60, 150), (85, 400), (255, 0, 0), 3)
    cv2.rectangle(img, (60, int(volBar)), (85, 400), (255, 0, 0), cv2.FILLED)
    cv2.putText(img, f'{int(volPer)} %', (40, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 3)
    cVol = int(volume.GetMasterVolumeLevelScalar() * 100)
    cv2.putText(img, f'Volume {int(cVol)}', (420, 70), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 18), 3)
    
    
    cTime = time.time()
    fps = 1/(cTime - pTime)
    pTime = cTime
    
    cv2.putText(img, f'Fps {str(int(fps))}', (10, 70), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 18), 3)
    cv2.imshow("Image", img)
    cv2.waitKey(1)
