import cv2
from ultralytics import YOLO
# from omegaconf import OmegaConf


### Load the YOLOv8 model
model = YOLO("./models/yolov8n-pose.pt")    

def RunOnVideoClip():

    ### Source image
    source = "./videos/video.mp4"

    ### "save = True" will create folders and will save the result video
    ### at "./runs/pose/predict/file_name"
    model.predict(source, save=True, imgsz=640, conf=0.5)



def RunOnWebcam():

    ### capture video camera (laptop)
    cap = cv2.VideoCapture(0)

    ### loop through the video frames
    while (cap.isOpened()):

        ### load a frame from the capture
        success, frame = cap.read()

        if success:
            
            ### run YOLO model and get the pose estimation data from the frame
            results = model.predict(frame, save=False, imgsz=640, conf=0.5)
            # print(results[0].keypoints.xyn)

            ### extract keypoint
            result_keypoint = results[0].keypoints.xyn.cpu().numpy()[0]
            # print(result_keypoint)

            ### visualize the results on the frame
            annotated_frame = results[0].plot()

            ### display the annotated frames
            cv2.imshow("YOLO v8 Pose", annotated_frame)

            if (cv2.waitKey(1) == ord('q')):
                break
        else:
            break

    ### Release the resources and close all windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':

    RunOnWebcam()
    