import cv2
from ultralytics import YOLO

CONF_LEVEL = 0.5

### Load the YOLOv8 model
model = YOLO("./models/yolov8n.pt")

### print out all object classes in pre-trained model
# print(model.names)

### method for model() results display
def getColors(color_num):
    base_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
    color_index = color_num % len(base_colors)
    increments = [(1, -2, 1), (-2, 1, -1), (1, -1, 2)]
    color = [base_colors[color_index][i] + increments[color_index][i] * 
            (color_num // len(base_colors)) % 256 for i in range(3)]
    return tuple(color)

### Main method
def RunYoloObjectDetection():

    ### capture web-cam videl (laptop)
    cap = cv2.VideoCapture(0)
   
    ### Loop through the video frames
    while (cap.isOpened()):

        ### Load a frame from the capture
        success, frame = cap.read()

        if success:
            # Detect person class: classes = 0
            # No output on screen: verbose = False
            
            ### detect all classes (1): model()
            results = model(frame, stream=True)
            
            ### detect all classes (2): model.detect()
            # results = model.predict(frame, save=False, conf=CONF_LEVEL, verbose=False)

            ### detect person class only
            # results = model.predict(frame, save=False, imgsz=640, conf=0.5, classes=0, verbose=False)
            # print(results)
            
            ### display detected objects by default 
            # for result in results:
            #     annotated_frame = result.plot()

            ### display detected objects by customized loop
            for result in results:
                class_names = result.names
                for box in result.boxes:
                    if(box.conf > CONF_LEVEL):
                        [x1, y1, x2, y2] = box.xyxy[0]
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                        class_id = int(box.cls[0])
                        # class_name = class_names[class_id]
                        class_color = getColors(class_id)

                        cv2.rectangle(frame, (x1, y1), (x2, y2), class_color, 2)
                        cv2.putText(frame, f'{class_names[int(box.cls[0])]} {box.conf[0]:.2f}', 
                                    (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, class_color, 2) 


            cv2.imshow("YOLO DETECTION", frame)

            if (cv2.waitKey(1) == ord('q')):
                break
        else:
            break

    ### Release the resources and close all windows
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':

    RunYoloObjectDetection() 
