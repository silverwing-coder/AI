'''
Ver. Apr-2-24
Created by Sangmork Park (Virginia Military Institute)
'''

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2

'''
@parameter: image
@return: face detect results
'''
def get_annotated_image_pose(rgb_image, detection_result):
  pose_landmarks_list = detection_result.pose_landmarks
  annotated_image = np.copy(rgb_image)

  # Loop through the detected poses to visualize.
  for idx in range(len(pose_landmarks_list)):
    pose_landmarks = pose_landmarks_list[idx]

    # Draw the pose landmarks.
    pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    pose_landmarks_proto.landmark.extend([
      landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in pose_landmarks
    ])
    solutions.drawing_utils.draw_landmarks(
      annotated_image,
      pose_landmarks_proto,
      solutions.pose.POSE_CONNECTIONS,
      solutions.drawing_styles.get_default_pose_landmarks_style())

  return annotated_image


'''
@parameter: image
@return: pose estimation landmarks results
'''
def get_pose_estimation_results(image):

    ''' image color channel affects detection performance
    convert color channel from GBR(Webcam) --> BGR (mediapipe) '''
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    base_options = python.BaseOptions(model_asset_path='pose_landmarker_lite.task')
    options = vision.PoseLandmarkerOptions(base_options=base_options, output_segmentation_masks=True)
    detector = vision.PoseLandmarker.create_from_options(options)

    ''' Load mediapipe-type input image '''
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)

    detection_result = detector.detect(image)

    return detection_result