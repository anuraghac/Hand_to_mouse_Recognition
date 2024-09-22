import cv2
import mediapipe as mp
import time
import pyautogui

def hand_detect():
    cap = cv2.VideoCapture(0)

    mpHands = mp.solutions.hands
    Hands = mpHands.Hands(max_num_hands = 1)
    mpDraw = mp.solutions.drawing_utils
    pyautogui.FAILSAFE = False

    ptime = 0
    ctime = 0

    while True:
        success, img = cap.read()
        img_height, img_width, _ = img.shape
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = Hands.process(imgRGB)
        # print(results.multi_hand_landmarks)

        if results.multi_hand_landmarks:
            for handLms in results.multi_hand_landmarks:

                thumb_tip = handLms.landmark[mpHands.HandLandmark.THUMB_TIP]
                ring_tip = handLms.landmark[mpHands.HandLandmark.RING_FINGER_TIP]

                # Check if the distance between thumb tip and index fingertip is small (fingers are touching)
                distance_threshold = 0.05
                distance_click = ((thumb_tip.x - ring_tip.x) ** 2 + (thumb_tip.y - ring_tip.y) ** 2) ** 0.5

                if distance_click < distance_threshold:
                    pyautogui.rightClick()

                thumb_tip = handLms.landmark[mpHands.HandLandmark.THUMB_TIP]
                middle_tip = handLms.landmark[mpHands.HandLandmark.MIDDLE_FINGER_TIP]

                # Check if the distance between thumb tip and index fingertip is small (fingers are touching)
                distance_threshold = 0.05
                distance_click = ((thumb_tip.x - middle_tip.x) ** 2 + (thumb_tip.y - middle_tip.y) ** 2) ** 0.5

                if distance_click < distance_threshold:
                    pyautogui.drag()

                thumb_tip = handLms.landmark[mpHands.HandLandmark.THUMB_TIP]
                index_tip = handLms.landmark[mpHands.HandLandmark.INDEX_FINGER_TIP]

                # Check if the distance between thumb tip and index fingertip is small (fingers are touching)
                distance_threshold = 0.05
                distance_click = ((thumb_tip.x - index_tip.x) ** 2 + (thumb_tip.y - index_tip.y) ** 2) ** 0.5

                if distance_click < distance_threshold:
                    pyautogui.click()


                for id, lm in enumerate(handLms.landmark):
                    # print(id,lm)
                    h, w, c = img.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    print(id, cx, cy)
                    if id == 8:
                        screen_width, screen_height = pyautogui.size()
                        target_x = int((cx / img_width) * screen_width)
                        target_y = int((cy / img_height) * screen_height)
                        pyautogui.moveTo(target_x, target_y)
                        cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)

                    if id == 4:

                        cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)

                mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

        ctime = time.time()
        fps = 1 / (ctime - ptime)
        ptime = ctime

        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

        cv2.imshow("Image", img)
        cv2.waitKey(1)


hand_detect()
