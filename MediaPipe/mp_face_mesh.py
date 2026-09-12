'''
Created by Sangmork Park on 1. September 2026
---------------------------------------------
Docstring for MediaPipe.runFaceDetect

'''

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import time

# --- Setup MediaPipe ---
model_path = '/home/parks/ml_python/models/face_landmarker.task' # Replace with your model path

def main():

    BaseOptions = mp.tasks.BaseOptions
    base_options = BaseOptions(model_asset_path=model_path, delegate=python.BaseOptions.Delegate.GPU)
    options = vision.FaceLandmarkerOptions(base_options=base_options, 
                                           output_face_blendshapes=True,
                                           output_facial_transformation_matrixes=True,
                                           num_faces=1,
                                        #    running_mode=vision.RunningMode.VIDEO
                                        )
    landmarker = vision.FaceLandmarker.create_from_options(options)

    # --- Webcam and Processing Loop ---
    cap = cv2.VideoCapture(0) # 0 for the default webcam

    if not cap.isOpened():
        print("Cannot open camera")
        exit()

    while cap.isOpened():
        success, frame = cap.read()

        if success:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

            lm_result = landmarker.detect(mp_image)

            if(lm_result.face_landmarks):
                for face_landmarks in lm_result.face_landmarks:
                    # Draw the face landmarks on the frame
                    for landmark in face_landmarks:
                        x = int(landmark.x * frame.shape[1])
                        y = int(landmark.y * frame.shape[0])
                        cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
                
            # Display the frame
            cv2.imshow('MediaPipe Face Mesh', cv2.flip(frame, 1))

            # Break loop with 'q' key
            if cv2.waitKey(5) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()