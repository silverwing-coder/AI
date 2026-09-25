'''
Docstring for MediaPipe.runFaceDetect

'''

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import PoseLandmarksConnections

import time
# import numpy as np
import math

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

def draw_face_landmarks(frame, face_landmarks):
    if face_landmarks:
        for detection in face_landmarks:
            bbox = detection.bounding_box
            x_min = int(bbox.origin_x)
            y_min = int(bbox.origin_y)
            width = int(bbox.width)
            height = int(bbox.height)

            # Draw bounding box on the frame
            cv2.rectangle(frame, (x_min, y_min), (x_min + width, y_min + height), (0, 0, 255), 2)

            # Loop and plot 6 keypoints on the face
            # 0-Right eye, 1-Left eye, 2-Nose tip, 3-Mouth center, 4-Right ear tragion, 5-Left ear tragion
            for keypoint in detection.keypoints:
                x = int(keypoint.x * frame.shape[1])
                y = int(keypoint.y * frame.shape[0])
                cv2.circle(frame, (x, y), 3, (0, 255, 255), -1)

        return frame

def check_facing_forward(face_landmarks):
    size_normalization_factor = 1000  # Adjust this factor based on your needs
    if face_landmarks:
        for detection in face_landmarks:
            # Get the x-coordinates of the right and left ear tragions
            right_ear_x = int(detection.keypoints[4].x * size_normalization_factor)  # Right ear tragion
            left_ear_x = int(detection.keypoints[5].x * size_normalization_factor)   # Left ear tragion
            nose_tip_x = int(detection.keypoints[2].x * size_normalization_factor)    # Nose tip

            right_ear_x_dst = abs(nose_tip_x - right_ear_x)
            left_ear_x_dst = abs(nose_tip_x - left_ear_x)
            max_ear_x_dst = max(right_ear_x_dst, left_ear_x_dst)
            threshold = 0.2 # 30% of the normalized frame width

            if right_ear_x_dst/max_ear_x_dst  < threshold or left_ear_x_dst/max_ear_x_dst < threshold:
                return False
            else:
                return True
    else:
        # If no face landmarks are provided, return False
        return False

def check_wearing_yellow_jacket(frame, pose_landmarks):
    # pass
    LEFT_SHOULDER_IDX = 11
    RIGHT_SHOULDER_IDX = 12 
    LEFT_HIP_IDX = 23
    RIGHT_HIP_IDX = 24

    h, w, _ = frame.shape
    landmarks = pose_landmarks[0] if pose_landmarks else None
    if landmarks is None:
        return False
    else:
        left_shoulder = landmarks[LEFT_SHOULDER_IDX]
        right_shoulder = landmarks[RIGHT_SHOULDER_IDX]
        left_hip = landmarks[LEFT_HIP_IDX]
        right_hip = landmarks[RIGHT_HIP_IDX]

        # Convert normalized coordinates to pixel coordinates
        left_shoulder_pixel = (int(left_shoulder.x * w), int(left_shoulder.y * h))
        right_shoulder_pixel = (int(right_shoulder.x * w), int(right_shoulder.y * h))
        left_hip_pixel = (int(left_hip.x * w), int(left_hip.y * h))
        right_hip_pixel = (int(right_hip.x * w), int(right_hip.y * h))

        # Define the bounding box for the torso region
        x_min = min(left_shoulder_pixel[0], right_shoulder_pixel[0], left_hip_pixel[0], right_hip_pixel[0])
        x_max = max(left_shoulder_pixel[0], right_shoulder_pixel[0], left_hip_pixel[0], right_hip_pixel[0])
        y_min = min(left_shoulder_pixel[1], right_shoulder_pixel[1], left_hip_pixel[1], right_hip_pixel[1])
        y_max = max(left_shoulder_pixel[1], right_shoulder_pixel[1], left_hip_pixel[1], right_hip_pixel[1])

        # Ensure the bounding box is within the frame dimensions
        x_min = max(0, x_min)
        y_min = max(0, y_min)
        x_max = min(w, x_max)
        y_max = min(h, y_max)

        # if frame is not None:
        #     cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)

        # Crop the torso region from the frame
        torso_region = frame[y_min:y_max, x_min:x_max]

        # Convert the cropped region to HSV color space
        hsv_torso_region = cv2.cvtColor(torso_region, cv2.COLOR_BGR2HSV)

        # Define the HSV range for yellow color
        # lower_yellow = (15, 100, 100)
        # upper_yellow = (35, 255, 255)        
  
        # Define the HSV range for "gray-blue" color (adjusted for better detection)
        lower_yellow = (95, 25, 75)
        upper_yellow = (120, 89, 255)        

        # Create a mask for yellow color in the torso region
        yellow_mask = cv2.inRange(hsv_torso_region, lower_yellow, upper_yellow)
        # cv2.imshow("Yellow Mask", yellow_mask)

        # yellow_pixel_count = cv2.countNonZero(yellow_mask)
        # total_pixels = (y_max - y_min) * (x_max - x_min)
        # total_pixels = (y_max - y_min) * (x_max - x_min)
        # print("Yellow Pixels:", yellow_pixel_count)
        # print("Total Pixels:", total_pixels)

        # Calculate the percentage of yellow pixels in the torso region
        yellow_percentage = (cv2.countNonZero(yellow_mask) / (torso_region.size / 3)) * 100  # Divide by 3 for the number of channels
        # print(yellow_percentage)

        # Define a threshold for determining if the person is wearing a yellow jacket
        yellow_threshold = 30  # Adjust this threshold based on your needs

        return yellow_percentage > yellow_threshold

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
    face_detect_model_path = '/home/parks/Documents/GestureComm/blaze_face_full_range.tflite' # Replace with your model path

    BaseOptions = mp.tasks.BaseOptions
    pose_base_options = BaseOptions(model_asset_path=pose_model_path, delegate=python.BaseOptions.Delegate.GPU)
    pose_options = vision.PoseLandmarkerOptions(base_options=pose_base_options, min_pose_detection_confidence=0.8)
    pose_detector = vision.PoseLandmarker.create_from_options(pose_options)

    face_base_options = BaseOptions(model_asset_path=face_detect_model_path, delegate=python.BaseOptions.Delegate.GPU)
    face_options = vision.FaceDetectorOptions(base_options=face_base_options)
    face_detector = vision.FaceDetector.create_from_options(face_options)   

    # --- Webcam and Processing Loop ---
    camera = cv2.VideoCapture(0) # 0 for the default webcam

    if not camera.isOpened():
        print("Cannot open camera")
        exit()

    previous_time = 0
    while camera.isOpened():
        success, frame = camera.read()

        if success:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

            """ POSE DETECTION """
            pose_result = pose_detector.detect(mp_image)
            pose_landmarks = pose_result.pose_landmarks

            """ Check if the person is wearing a yellow jacket based on the pose landmarks"""
            is_wearing_yellow_jacket = check_wearing_yellow_jacket(frame, pose_landmarks)

            if is_wearing_yellow_jacket:
                cv2.putText(frame, "Wearing Yellow Jacket", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            else:
                cv2.putText(frame, "Not Wearing Yellow Jacket", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                continue  # Skip the rest of the loop and go to the next frame
            # print("Is wearing yellow jacket:", is_wearing_yellow_jacket)

            # draw pose landmarks on the frame
            frame = draw_pose_landmarks(frame, pose_landmarks)

            """ FACE DETECTION """
            face_detection_result = face_detector.detect(mp_image)
            face_landmarks = face_detection_result.detections

            if face_landmarks is None or len(face_landmarks) == 0:
                print("No face detected in this frame.")
                # cv2.putText(frame, "No Face Detected", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                # continue  # Skip the rest of the loop and go to the next frame
            else:
                if frame is None:
                    print("Frame is None, skipping this iteration.")
                    continue         

                frame = draw_face_landmarks(frame, face_landmarks)
                # print("Face detected in this frame.")
                # cv2.putText(frame, "Face Detected", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                is_facing_forward = check_facing_forward(face_landmarks)
                if is_facing_forward:
                    cv2.putText(frame, "Facing Forward", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                else:
                    cv2.putText(frame, "Not Facing Forward", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                # print(is_facing_forward)

                #if is_facing_forward == True:
                # # is_wearing_yellow_jacket = is_wearing_yellow_jacket(frame)
                # # if is_facing_forward == True and is_wearing_yellow_jacket == True:
                #     cv2.putText(frame, "Continue Processing", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                # else:
                #     cv2.putText(frame, "Stop Processing", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                #     continue  # Skip the rest of the loop and go to the next frame
                
            current_time = time.time()
            fps = 1 / (current_time - previous_time)
            previous_time = current_time
            cv2.putText(frame, f'FPS: {int(fps)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            # if frame is None:
            #     print("Frame is None, skipping this iteration.")
            #     continue
            if frame is not None:
                cv2.imshow('GESTURE COMM', frame)

            # Break loop with 'q' key
            if cv2.waitKey(5) & 0xFF == 27:
                break

    camera.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()

    # p1 = (50, 50)
    # # p2 = (100, 100)
    # p2 = (50, 100)
    # # p2 = (0, 100)

    # print(f"Angle and Distance between {p1} and {p2}:", get_angle_dst_between_points(p1, p2))
 