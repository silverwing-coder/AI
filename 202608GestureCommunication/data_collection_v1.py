'''
Created by Sangmork Park at VMI
Last updated: 09-20-2026

This script captures video from the webcam, detects human pose landmarks using MediaPipe, 
calculates angles and distances between nose tip and specific landmarks(shoulders, elbows, and wrists), 
and saves the data to a CSV file for gesture recognition tasks.

'''

"""
$ pip install mediapipe
$ pip install pandas
"""

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import PoseLandmarksConnections

import time
import math

import pandas as pd

'''
CLASS:1 ~ 10. 
Each classes describe a gesture for communication.   
'''
CLASS = 10

def draw_pose_landmarks(frame, pose_landmarks):
    
    if pose_landmarks:
        landmarks = pose_landmarks[0]
        for connection in PoseLandmarksConnections.POSE_LANDMARKS:
            start_idx, end_idx = connection.start, connection.end
            start_landmark = landmarks[start_idx]
            end_landmark = landmarks[end_idx]
            start_x = int(start_landmark.x * frame.shape[1])
            start_y = int(start_landmark.y * frame.shape[0])
            end_x = int(end_landmark.x * frame.shape[1])
            end_y = int(end_landmark.y * frame.shape[0])

            cv2.line(frame, (start_x, start_y), (end_x, end_y), (0, 255, 0), 2)

        return frame

def get_angle_dst_between_points(p1, p2):
    """
    Calculate the angle between three points (p1, p2) in degrees.
    """
    angle = math.degrees(math.atan2(p2[1] - p1[1], p2[0] - p1[0])) % 360
    distance = math.dist(p1, p2)
    
    return angle, distance



def main(): 

    # --- Setup MediaPipe ---
    pose_model_path = '/home/parks/Documents/GestureComm/pose_landmarker_full.task' # Replace with your model path

    options = vision.PoseLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=pose_model_path), # delegate=python.BaseOptions.Delegate.GPU),
        running_mode=vision.RunningMode.VIDEO
    )

    # --- Webcam and Processing Loop ---
    camera = cv2.VideoCapture(0) # 0 for the default webcam

    if not camera.isOpened():
        print("Cannot open camera")
        exit()

    previous_time = 0
    data_count = 0

    # Define your column names (12 features + 1 label)
    columns = [
        'angle_1', 'dist_1', 'angle_2', 'dist_2', 'angle_3', 'dist_3',
        'angle_4', 'dist_4', 'angle_5', 'dist_5', 'angle_6', 'dist_6', 
        'label'
    ]
    while camera.isOpened():
        success, frame = camera.read()

        if success:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

            with vision.PoseLandmarker.create_from_options(options) as pose_detector:

                """ POSE DETECTION """
                # pose_result = pose_detector.detect(mp_image)
                pose_result = pose_detector.detect_for_video(mp_image, int(time.time() * 1000))
                pose_landmarks = pose_result.pose_landmarks

                # draw pose landmarks on the frame
                frame = draw_pose_landmarks(frame, pose_landmarks)

                # pose_landmarks = pose_landmarks[0]

                LEFT_SHOULDER_IDX = 11
                RIGHT_SHOULDER_IDX = 12 

                if frame is not None:
                    h, w, _ = frame.shape
             
                if pose_landmarks:
                    # landmarks = pose_landmarks[0]
                    left_shoulder = pose_landmarks[0][LEFT_SHOULDER_IDX]
                    right_shoulder = pose_landmarks[0][RIGHT_SHOULDER_IDX]
                    shoulder_line = int((left_shoulder.y * h + right_shoulder.y * h) / 2)
                    nose_tip = pose_landmarks[0][0]  # Assuming the nose tip is the first landmark
                    distance_from_nose_to_shoulder = abs(nose_tip.y * h - shoulder_line)
                    print(f'Distance from Nose to Shoulder: {int(distance_from_nose_to_shoulder)}')

                    cv2.circle(frame, (int(nose_tip.x * w), int(nose_tip.y * h)), 20, (0, 255, 255), -1)

                    raw_data = []  # Initialize raw_datalist
                    for idx in range(11, 17):  # Loop through left and right shoulders, elbows, and wrists
                        landmark = pose_landmarks[0][idx]
                        # print(landmark)
                        x = int(landmark.x * w)
                        y = int(landmark.y * h)
                        cv2.circle(frame, (x, y), 10, (0, 0, 255), -1)

                        angle, distance = get_angle_dst_between_points((nose_tip.x * w, nose_tip.y * h), (x, y))
                        normalized_distance = distance / distance_from_nose_to_shoulder if distance_from_nose_to_shoulder != 0 else 0

                        cv2.putText(frame, f'Angle: {int(angle)}', (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)
                        cv2.putText(frame, f'Distance: {normalized_distance:.2f}', (x + 10, y + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)

                        # raw_data.append(f"{angle:.4f}")
                        # raw_data.append(f"{normalized_distance:.4f}")
                        raw_data.append(angle)
                        raw_data.append(normalized_distance)
                        
                    data_count += 1
                    raw_data.append(CLASS)  # Append the class label to the raw data
                    # print(f'Save data: {data_count}')
                    # print(f'Raw Data: {raw_data}')
                    
                    df = pd.DataFrame([raw_data], columns=columns)
                    # Append to CSV without writing the header if the file already exists
                    df.to_csv('gesture_data.csv', mode='a', header=not pd.io.common.file_exists('gesture_data.csv'), index=False)

            current_time = time.time()
            fps = 1 / (current_time - previous_time)
            previous_time = current_time
            cv2.putText(frame, f'FPS: {int(fps)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.imshow('GESTURE COMM', frame)


            # Break loop with 'ESC' key
            if cv2.waitKey(1) & 0xFF == 27:
                break

            """
            The statements below saves data and save an image for each classes. 
            /** Uncomment only when collect data **/
            """
            # if data_count == 50:
            #     cv2.imwrite(f'gesture_data_{CLASS}.jpg', frame)
            
            # if data_count > 100:
            #     break

    camera.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()


 