# import os
# os.environ["QT_FONTCONFIG"] = "1"

import cv2
import mediapipe as mp

camera = cv2.VideoCapture(0)

while camera.isOpened():

    success, frame = camera.read()

    if success:
        cv2.imshow("'CAM'", frame)


    if cv2.waitKey(1) == 27:
        break

cv2.destroyAllWindows()
camera.release()