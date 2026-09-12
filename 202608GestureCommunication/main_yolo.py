import cv2
from ultralytics import YOLO
import numpy as np

import helper_functions as hf

POSE_CONFIDENCE_THRESHOLD = 0.7 # threshold value for object detection (person)
def main():
    camera = cv2.VideoCapture(0)
    model = YOLO('yolo26n-pose.pt')

    while camera.isOpened():
        ret, frame = camera.read()
        if ret:
            results = model(frame, stream=True)
            for result in results:
                ''' filter out with confidence threshold: 0.7 '''
                mask = result.boxes.conf >= POSE_CONFIDENCE_THRESHOLD
                # print(mask)
                result.boxes = result.boxes[mask]
                result.keypoints = result.keypoints[mask]

                annotated_frame = result.plot()

                # 1. Crop the boxed-frame (person) from annotated_frame
                boxes = result.boxes.xyxy.cpu().numpy()
                for idx, box in enumerate(boxes):
                    x_min, y_min, x_max, y_max = map(int, box)
                    person_crop = annotated_frame[y_min:y_max, x_min:x_max]

                if result.keypoints is not None:
                    keypoints = result.keypoints.xy.cpu().numpy()
                    for idx, (box, kp) in enumerate(zip(boxes, keypoints)):
                        # /* Keypoints: YOLO 26
                        # 0-nose, 1-right eye, 2-right eye, 3-left ear, 4-right ear, 5-left shoulder, 6- right shoulder
                        # 7-left elbow, 8-right elbow, 9-left wrist, 10-right wrist, 11-left hip, 12-right hip,
                        # 13-left knee, 14-right-knee, 15-left ankle, 16-right ankle
                        x_min, y_min, x_max, y_max = map(int, box)

                        # 2. crop the face (head-top to shoulders) from annotated_frame
                        #    cropped face image is used to identify if the person is wearing a colored jacket
                        left_shoulder_y = kp[5][1]
                        right_shoulder_y = kp[6][1]
                        if left_shoulder_y == 0 and right_shoulder_y == 0:
                            continue
                        # Determine the lower bound (y_max) for the face crop using shoulders
                        # Filter out undetected individual shoulders if one is 0
                        valid_shoulders = [y for y in [left_shoulder_y, right_shoulder_y] if y > 0]
                        shoulder_y_cutoff = int(np.mean(valid_shoulders))

                        nose_y = int(kp[0][1]) if kp[0][1] > 0 else 0
                        head_top_y = max(nose_y - (shoulder_y_cutoff - nose_y), 0)

                        # Ensure the shoulder cutoff is logically within the bounding box
                        if shoulder_y_cutoff > y_min:
                            nose_x = int(kp[0][0]) if kp[0][0] > 0 else 0
                            nose_y = int(kp[0][1]) if kp[0][1] > 0 else 0
                            # head_top_y = max(nose_y - (shoulder_y_cutoff - nose_y), 0)
                            head_x_min = max(nose_x - shoulder_y_cutoff//2, 0)
                            head_x_max = min(nose_x + shoulder_y_cutoff//2, x_max)

                            # Crop from top of bounding box down to the shoulder line
                            # face_crop = annotated_frame[y_min:shoulder_y_cutoff, x_min:x_max]
                            # face_crop = annotated_frame[head_top_y:shoulder_y_cutoff, x_min:x_max]
                            # face_crop = annotated_frame[head_top_y:shoulder_y_cutoff, head_x_min:head_x_max]
                            face_crop = frame[head_top_y:shoulder_y_cutoff, head_x_min:head_x_max]
                            # face_crop = frame[y_min:shoulder_y_cutoff, x_min:x_max]


                        # 2-1. normalize face_crop image with fixed height (200 pixels)
                        target_height = 200
                        org_height, org_width = face_crop.shape[:2]
                        if org_height == 0:
                            continue
                        aspect_ratio = org_width / org_height
                        target_width = int(target_height * aspect_ratio)
                        resized_face = cv2.resize(face_crop, (target_width, target_height), interpolation=cv2.INTER_AREA)

                        is_facing_forward = hf.is_facing_forward(resized_face)
                        if is_facing_forward:
                            cv2.putText(annotated_frame, "Facing Forward", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 1)
                            # cv2.putText(frame, "Facing Forward", (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                        else:
                            cv2.putText(annotated_frame, "Not Facing Forward", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 1)
                            # cv2.putText(frame, "Not Facing Forward", (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

                    # 3. crop the body (shoulders to hips) from annotated_frame
                    #    cropped body image is used to identify if the person is facing forward
                    for idx, kp in enumerate(keypoints):
                        # Check if essential keypoints were detected (confidence > 0)
                        # COCO indices: 5=L_Shoulder, 6=R_Shoulder, 11=L_Hip, 12=R_Hip
                        if any(kp[idx][0] == 0 and kp[idx][1] == 0 for idx in [5, 6, 11, 12]):
                            continue
                        left_shoulder, right_shoulder = kp[5], kp[6]
                        left_hip, right_hip = kp[11], kp[12]

                        # Y bounds: from the highest shoulder to the lowest hip
                        y_body_min = int(min(left_shoulder[1], right_shoulder[1]))
                        y_body_max = int(max(left_hip[1], right_hip[1]))

                        # X bounds: extreme outer edges of the shoulders and hips
                        x_body_min = int(min(left_shoulder[0], right_shoulder[0], left_hip[0], right_hip[0]))
                        x_body_max = int(max(left_shoulder[0], right_shoulder[0], left_hip[0], right_hip[0]))

                        # 5. Add a small optional margin (padding) to prevent tight cutoff
                        h_pad = int((y_body_max - y_body_min) * 0.05)
                        w_pad = int((x_body_max - x_body_min) * 0.05)

                        img_h, img_w, _ = frame.shape
                        y1 = max(0, y_body_min - h_pad)
                        y2 = min(img_h, y_body_max + h_pad)
                        x1 = max(0, x_body_min - w_pad)
                        x2 = min(img_w, x_body_max + w_pad)

                        body_crop = annotated_frame[y1:y2, x1:x2]

                    # 4. crop the sign image (top-hands to hips) from annotated_frame
                    #    cropped sign image is used to collect data for machine learning application
                    for idx, kp in enumerate(keypoints):
                        # Check if essential keypoints were detected (confidence > 0)
                        # COCO indices: 5=L_Shoulder, 6=R_Shoulder, 11=L_Hip, 12=R_Hip
                        if any(kp[idx][0] == 0 and kp[idx][1] == 0 for idx in range(5, 12)):
                            continue
                        # left_shoulder, right_shoulder = kp[5], kp[6]
                        left_hip, right_hip = kp[11], kp[12]

                        # Y bounds: from the highest shoulder to the lowest hip
                        # y_min = int(min(left_shoulder[1], right_shoulder[1]))
                        y_sign_max = int(max(left_hip[1], right_hip[1]))

                        sign_crop = annotated_frame[y_min:y_sign_max, x_min:x_max]

                        # 4-1. normalize sign_crop image with fixed size (200 pixels)
                        # target_width = 200
                        target_height = 400
                        org_height, org_width = sign_crop.shape[:2]
                        if org_height == 0:
                            continue
                        aspect_ratio = org_width / org_height
                        target_width = int(target_height * aspect_ratio)
                        resized_sign = cv2.resize(sign_crop, (target_width, target_height), interpolation=cv2.INTER_AREA)

            # cv2.imshow("Gesture-based Communication", cv2.flip(annotated_frame, 1))
            # cv2.imshow("PERSON CROP", cv2.flip(person_crop, 1))
            # cv2.imshow("FACE CROP", cv2.flip(face_crop, 1))
            # cv2.imshow("FACE CROP", cv2.flip(resized_face, 1))
            # cv2.imshow("BODY CROP", cv2.flip(body_crop, 1))
            # cv2.imshow("SIGN CROP", cv2.flip(sign_crop, 1))
            # cv2.imshow("SIGN CROP", cv2.flip(resized_sign, 1))
            cv2.imshow("Gesture-based Communication", annotated_frame)

        if cv2.waitKey(1) & 0xFF == 27:     # 27: Escape-key
            break

    camera.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()