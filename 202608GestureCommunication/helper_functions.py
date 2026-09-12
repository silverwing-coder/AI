# Created by Sangmork Park at VMI on Sep. 2026

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

def is_facing_forward(face_image):
    """
    Determines if the face in the given image is facing forward based on the positions of key facial landmarks.
    Args:
        face_image (numpy.ndarray): The input image containing a face.
    Returns:
        bool: True if the face is facing forward, False otherwise.
    """

    # --- Setup MediaPipe ---
    model_path = '/home/parks/Documents/GestureComm/blaze_face_full_range.tflite' # Replace with your model path
    BaseOptions = mp.tasks.BaseOptions
    base_options = BaseOptions(model_asset_path=model_path, delegate=python.BaseOptions.Delegate.GPU)
    options = vision.FaceDetectorOptions(base_options=base_options)
    detector = vision.FaceDetector.create_from_options(options)

    # Convert the input image to RGB format
    rgb_frame = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    detection_result = detector.detect(mp_image)

    if detection_result.detections:
        # Assuming only one face is detected, get the first detection
        detection = detection_result.detections[0]

        # Get the keypoints for the detected face
        keypoints = detection.keypoints

        # Extract the x-coordinates distance from nose tip to two ears respectively
        # nose_tip: keypoints[2], right_ear: keypoints[4], left_ear: keypoints[5]
        right_ear_x_dst = abs(keypoints[2].x - keypoints[4].x) * face_image.shape[1]
        left_ear_x_dst = abs(keypoints[2].x - keypoints[5].x) * face_image.shape[1]
        max_x_distance = max(right_ear_x_dst, left_ear_x_dst)

        # Calculate the horizontal distance difference between the ears
        dst_difference = abs(right_ear_x_dst - left_ear_x_dst)

        # Define a threshold for determining if the face is facing forward
        threshold = 0.50 * max_x_distance  # 50% of the maximum distance between the nose and the ears

        # If the distance between ears is greater than the threshold, consider it not facing forward
        if dst_difference > threshold:
            return False
        else:
            return True
    else:
        # No face detected, return False
        return False

def is_wearing_yellow_jacket(body_image):
    """
    Determines if the person in the given image is wearing a yellow jacket based on color detection.
    Args:
        body_image (numpy.ndarray): The input image containing a person's upper body.
    Returns:
        bool: True if the person is wearing a yellow jacket, False otherwise."""
    
    # Convert the image to HSV color space
    hsv_image = cv2.cvtColor(body_image, cv2.COLOR_BGR2HSV)

    # Define the range for yellow color in HSV
    lower_yellow = (20, 100, 100)
    upper_yellow = (30, 255, 255)

    # Create a mask for yellow color
    yellow_mask = cv2.inRange(hsv_image, lower_yellow, upper_yellow)

    # Calculate the percentage of yellow pixels in the mask
    yellow_pixel_count = cv2.countNonZero(yellow_mask)
    total_pixel_count = body_image.shape[0] * body_image.shape[1]
    yellow_percentage = (yellow_pixel_count / total_pixel_count) * 100

    # Define a threshold for determining if the person is wearing a yellow jacket
    threshold_percentage = 70.0  # Adjust this value based on your requirements

    # If the percentage of yellow pixels is greater than the threshold, consider it as wearing a yellow jacket
    if yellow_percentage > threshold_percentage:
        return True
    else:
        return False
