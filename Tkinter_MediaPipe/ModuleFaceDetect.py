'''
Ver. Apr-2-24
Created by Sangmork Park (Virginia Military Institute)
'''

import cv2
import math
import mediapipe as mp
from typing import Tuple, Union
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

MARGIN = 10  # pixels
ROW_SIZE = 10  # pixels
FONT_SIZE = 1.5
FONT_THICKNESS = 2
TEXT_COLOR = (255, 255, 255)
BOX_COLOR = (0, 255, 0)


def _normalized_to_pixel_coordinates(

    normalized_x: float, normalized_y: float, image_width: int,
    image_height: int) -> Union[None, Tuple[int, int]]:

  '''   Converts normalized value pair to pixel coordinates.
        Checks if the float value is between 0 and 1. '''
  def is_valid_normalized_value(value: float) -> bool:
    return (value > 0 or math.isclose(0, value)) and (value < 1 or
                                                      math.isclose(1, value))

  if not (is_valid_normalized_value(normalized_x) and
          is_valid_normalized_value(normalized_y)):
    ''' TODO: Draw coordinates even if it's outside of the image bounds '''
    return None
  x_px = min(math.floor(normalized_x * image_width), image_width - 1)
  y_px = min(math.floor(normalized_y * image_height), image_height - 1)
  return x_px, y_px

'''
@parameter: image, faces detected
@return: annotated image
Draws bounding boxes and keypoints on the input image and returns it
'''
def get_annotated_image_faces(image, results):

    annotated_image = image.copy()
    height, width, _ = image.shape

    for detection in results.detections:
        # Draw bounding_box
        bbox = detection.bounding_box
        start_point = bbox.origin_x, bbox.origin_y
        end_point = bbox.origin_x + bbox.width, bbox.origin_y + bbox.height
        cv2.rectangle(annotated_image, start_point, end_point, BOX_COLOR, 3)

        ''' Draw keypoints '''
        for keypoint in detection.keypoints:
            keypoint_px = _normalized_to_pixel_coordinates(keypoint.x, keypoint.y,
                                                           width, height)
            color, thickness, radius = (0, 255, 0), 2, 2
            cv2.circle(annotated_image, keypoint_px, thickness, color, radius)

        ''' Draw label and score '''
        category = detection.categories[0]
        category_name = category.category_name
        category_name = '' if category_name is None else category_name
        probability = round(category.score, 2)
        result_text = category_name + ' (' + str(probability) + ')'
        text_location = (MARGIN + bbox.origin_x,
                         MARGIN + ROW_SIZE + bbox.origin_y)
        cv2.putText(annotated_image, result_text, text_location, cv2.FONT_HERSHEY_PLAIN,
                    FONT_SIZE, TEXT_COLOR, FONT_THICKNESS)

    return annotated_image

'''
@parameter: image
@return: faces detection results
'''
def get_face_detected_results(image):

    ''' color channel affects the performance (BGR -> RGB) '''
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    base_options = python.BaseOptions(model_asset_path='blaze_face_short_range.tflite')
    options = vision.FaceDetectorOptions(base_options=base_options)
    detector = vision.FaceDetector.create_from_options(options)

    ''' Load mediapipe-type input image '''
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)

    detection_result = detector.detect(mp_image)

    return detection_result
