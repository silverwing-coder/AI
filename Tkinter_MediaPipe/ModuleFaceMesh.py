'''
Ver. Apr-2-24
Created by Sangmork Park (Virginia Military Institute)
'''

import cv2
import numpy as np
import mediapipe as mp
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

'''
@parameter: image, face landmarks
@return: annotated image
'''
def get_annotated_image_facemesh(rgb_image, detection_result):

  face_landmarks_list = detection_result.face_landmarks
  annotated_image = np.copy(rgb_image)

  ''' setup drawing spec: circle size, line-color, line-thickness, etc.'''
  tesslatin_drawing_spec = mp.solutions.drawing_utils.DrawingSpec(thickness=0, circle_radius=1, color=(0, 255, 255))
  # contour_drawing_spec = mp.solutions.drawing_utils.DrawingSpec(color=(255, 255, 255))

  ''' Loop through the detected faces landmarks '''
  for idx in range(len(face_landmarks_list)):
    face_landmarks = face_landmarks_list[idx]

    face_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    face_landmarks_proto.landmark.extend([
      landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in face_landmarks
    ])

    solutions.drawing_utils.draw_landmarks(
        image=annotated_image,
        landmark_list=face_landmarks_proto,
        connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
        landmark_drawing_spec=tesslatin_drawing_spec,
        connection_drawing_spec=mp.solutions.drawing_styles
        .get_default_face_mesh_tesselation_style())
    # solutions.drawing_utils.draw_landmarks(
    #     image=annotated_image,
    #     landmark_list=face_landmarks_proto,
    #     connections=mp.solutions.face_mesh.FACEMESH_CONTOURS,
    #     landmark_drawing_spec=contour_drawing_spec,
    #     connection_drawing_spec=mp.solutions.drawing_styles
    #     .get_default_face_mesh_contours_style())

    solutions.drawing_utils.draw_landmarks(
        image=annotated_image,
        landmark_list=face_landmarks_proto,
        connections=mp.solutions.face_mesh.FACEMESH_IRISES,
          landmark_drawing_spec=None,
          connection_drawing_spec=mp.solutions.drawing_styles
          .get_default_face_mesh_iris_connections_style())

  return annotated_image

'''
@parameter: image
@return: face landmarks
'''
def get_facemesh_results(image):

    ''' image color channel affects detection performance
    convert color channel from GBR(Webcam) --> BGR (mediapipe) '''
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    base_options = python.BaseOptions(model_asset_path='face_landmarker.task')
    # base_options = python.BaseOptions(model_asset_path='face_landmarker_v2_with_blendshapes.task')
    options = vision.FaceLandmarkerOptions(base_options=base_options,
                                           output_face_blendshapes=False,
                                           output_facial_transformation_matrixes=True,
                                           num_faces=1)
    detector = vision.FaceLandmarker.create_from_options(options)

    ''' Load mediapipe-type input image '''
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)

    detection_result = detector.detect(mp_image)

    return detection_result