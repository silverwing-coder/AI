import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import time

# --- Setup MediaPipe ---
model_path = '/home/sangmork/ml-python/models/face_landmarker.task' # Replace with your model path

import cv2
import time

import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks import python

BaseOptions = mp.tasks.BaseOptions
# HandLandmarker = mp.tasks.vision.HandLandmarker
# HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
# HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult
# VisionRunningMode = mp.tasks.vision.RunningMode

def draw_landmarks_on_image_manually(rgb_image, detection_result):

    HAND_CONNECTIONS = frozenset([
    (0, 1), (1, 2), (2, 3), (3, 4),             # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),             #Index finger
    (5, 9), (9, 10), (10, 11), (11, 12),        # Middle finger
    (9, 13), (13, 14), (14, 15), (15, 16),      # Ring finger
    (13, 17), (17, 18), (18, 19), (19, 20),     # Pinky finger
    (0, 17)                                     # Wrist to pinky base
    ])

    if not detection_result:
        return rgb_image
    
    img_height, img_width = rgb_image.shape[:2]
    # rgb_image = cv2.flip(rgb_image, 1)
    
    if detection_result.hand_landmarks:
        # print(detection_result)
        # print(detection_result.handedness)
        # print(detection_result.hand_landmarks)
        # print(detection_result.hand_landmarks[0])        
        # print(detection_result.hand_landmarks[0][0])

        all_landmarks_location = []
        all_handedness = []
        # for landmarks in detection_result.hand_landmarks:
        for i in range(len(detection_result.hand_landmarks)):
            landmarks = detection_result.hand_landmarks[i]
            handedness = detection_result.handedness[i]
            
            all_handedness.append(handedness[0].display_name)
            # print(handedness[0].display_name)

            one_hand_landmarks = []
            all_landmarks_location.append(one_hand_landmarks)
            for landmark in landmarks:
                
                # landmark.x, landmark.y: normailized landmarks location
                px, py = int(landmark.x * img_width), int(landmark.y * img_height)
                cv2.circle(rgb_image, (px, py), 5, (255, 255, 0), cv2.FILLED)
                one_hand_landmarks.append([px, py])
        
        for i in range(len(all_landmarks_location)):
            one_hand_landmarks = all_landmarks_location[i]
            for connection in HAND_CONNECTIONS:
                point1 = one_hand_landmarks[connection[0]]
                point2 = one_hand_landmarks[connection[1]]
                cv2.line(rgb_image, point1, point2, (0, 255, 0), 2)

            handedness = all_handedness[i]
            cv2.putText(rgb_image, handedness, one_hand_landmarks[0], cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)


    return rgb_image


def main():

    # model_path = '/home/parks/ml_python/models/hand_landmarker.task'
    # model_path = '/home/parks/ml_python/models/hand_landmarks_detector.tflite'
    model_path = '/home/parks/ml_python/models/hand_detector.tflite'

    # base_options=BaseOptions(model_asset_path=model_path, delegate=python.BaseOptions.Delegate.CPU)
    base_options=BaseOptions(model_asset_path=model_path, delegate=python.BaseOptions.Delegate.GPU)
    # options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=2)
    options = vision.HandLandmarkerOptions(base_options=base_options)
    detector = vision.HandLandmarker.create_from_options(options)

    capture = cv2.VideoCapture(0)
    
    while capture.isOpened():
        start = time.time()
        success, frame = capture.read()
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # rgb_frame = cv2.flip(rgb_frame, 1)

        if success:

            # image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            detection_result = detector.detect(mp_image)

            annotated_frame = draw_landmarks_on_image_manually(rgb_frame, detection_result)

            end = time.time()
            fps = round(1 / (end - start), 2)
            frame = cv2.putText(annotated_frame, str(fps), (20, 30), cv2.FONT_HERSHEY_COMPLEX, 0.7,
                        (255, 0, 255), 1)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            cv2.imshow('Hand Gesture', frame)

        if cv2.waitKey(1) == 27:
            break
        
    cv2.destroyAllWindows()
    capture.release()

if __name__ == '__main__':
  main()


