import cv2
import mediapipe as mp
import time

cam = cv2.VideoCapture(0)

mpHands = mp.solutions.hands
hands = mpHands.Hands()

while True:
    success, img = cam.read()

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    cv2.imshow("Image", img)
    cv2.waitKey(1)
