'''
Docstring for MediaPipe.runFaceDetect
'''

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import time

def main(): 

    # --- Setup MediaPipe ---
    face_detect_model_path = '/home/parks/Documents/GestureComm/blaze_face_full_range.tflite' # Replace with your model path
    BaseOptions = mp.tasks.BaseOptions
    base_options = BaseOptions(model_asset_path=face_detect_model_path, delegate=python.BaseOptions.Delegate.GPU)
    options = vision.FaceDetectorOptions(base_options=base_options)
    detector = vision.FaceDetector.create_from_options(options)

    # --- Webcam and Processing Loop ---
    camera = cv2.VideoCapture(0) # 0 for the default webcam

    if not camera.isOpened():
        print("Cannot open camera")
        exit()

    while camera.isOpened():
        success, frame = camera.read()

        if success:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

            detection_result = detector.detect(mp_image)
            # print(f"Faces detected: {len(detection_result.detections)}")

            if(detection_result.detections):
                height, width, _ = frame.shape
                for detection in detection_result.detections:
                    bbox = detection.bounding_box
                    x_min = int(bbox.origin_x)
                    y_min = int(bbox.origin_y)
                    width = int(bbox.width)
                    height = int(bbox.height)

                    # Draw bounding box on the frame
                    cv2.rectangle(frame, (x_min, y_min), (x_min + width, y_min + height), (0, 255, 0), 2)

                    # Loop and plot 6 keypoints on the face
                    # 0-Right eye, 1-Left eye, 2-Nose tip, 3-Mouth center, 4-Right ear tragion, 5-Left ear tragion
                    for keypoint in detection.keypoints:
                        x = int(keypoint.x * frame.shape[1])
                        y = int(keypoint.y * frame.shape[0])
                        cv2.circle(frame, (x, y), 3, (0, 255, 255), -1)
                    nose_x = int(detection.keypoints[2].x * frame.shape[1])
                    nose_y = int(detection.keypoints[2].y * frame.shape[0])
                    right_ear_x = int(detection.keypoints[4].x * frame.shape[1])
                    left_ear_x = int(detection.keypoints[5].x * frame.shape[1])

                    difference = abs(right_ear_x - nose_x) - abs(left_ear_x - nose_x)
                    difference = abs(right_ear_x - nose_x) / abs(left_ear_x - nose_x)
                    if difference > 50:
                        cv2.putText(frame, "Head turned to the left", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    elif difference < -50:
                        cv2.putText(frame, "Head turned to the right", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    else:
                        cv2.putText(frame, "Head is facing forward", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            #     print(detection_result)

            # Display the frame
            # cv2.imshow('MediaPipe Face Mesh', cv2.flip(frame, 1))
            cv2.imshow('MediaPipe Face Mesh', frame)

            # Break loop with 'q' key
            if cv2.waitKey(5) & 0xFF == ord('q'):
                break

    camera.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()