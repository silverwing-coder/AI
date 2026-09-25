''' function module for YOLOv8 models implementation
    @author: Sangmork Park (Virginia Military Institute) '''

''' Library modules '''
import tkinter as tk
from PIL import Image, ImageTk
import cv2
from ultralytics import YOLO

''' Customized modules '''
import FunctionsModule as fm # will be used for additional functionality
import YoloModelAppCalls as ym

class MainWindow:
    def __init__(self, window):
        # self.window = tk.Tk()
        self.window = window
        self.window.title("YOLOv8 on Tkinter")

        self.canvas_frame = tk.LabelFrame(self.window, text="Canvas Frame for Webcam Image")
        self.canvas_frame.grid(row=0, column=0, padx=10, pady=10)
        self.canvas = tk.Canvas(self.canvas_frame, width=640, height=480)
        self.canvas.pack()

        ''' Mode options frame '''
        self.options_frame = tk.Frame(self.window)
        # self.options_frame.grid(row=0, column=1, padx=10, pady=10, sticky="news")
        self.options_frame.grid(row=0, column=1, padx=10, pady=10, sticky="news")

        self.mode_frame = tk.LabelFrame(self.options_frame, text="Mode Options")
        self.mode_frame.grid(row=0, column=0, padx=10, pady=10, sticky="news")
        # self.mode_frame.grid(row=0, column=0, padx=10, pady=10)

        self.detection_var = tk.StringVar(value="Off")
        self.detection_option = tk.Checkbutton(self.mode_frame, text="Detection Mode", variable=self.detection_var, onvalue="On", offvalue="Off")
        self.detection_option.grid(row=1, column=0, padx=5, pady=5, sticky="W")

        # self.classification_var = tk.StringVar(value="Off")
        # self.classification_option = tk.Checkbutton(self.mode_frame, text="Classification Mode", variable=self.classification_var, onvalue="On", offvalue="Off")
        # self.classification_option.grid(row=2, column=0, padx=5, pady=5, sticky="W")
        #
        # self.segmentation_var = tk.StringVar(value="Off")
        # self.segmentation_option = tk.Checkbutton(self.mode_frame, text="Segmentation Mode", variable=self.segmentation_var, onvalue="On", offvalue="Off")
        # self.segmentation_option.grid(row=3, column=0, padx=5, pady=5, sticky="W")

        self.tracking_var = tk.StringVar(value="Off")
        self.tracking_option = tk.Checkbutton(self.mode_frame, text="Tracking Mode", variable=self.tracking_var, onvalue="On", offvalue="Off")
        self.tracking_option.grid(row=4, column=0, padx=5, pady=5, sticky="W")

        self.pose_var = tk.StringVar(value="Off")
        self.pose_option = tk.Checkbutton(self.mode_frame, text="Pose Mode", variable=self.pose_var, onvalue="On", offvalue="Off")
        self.pose_option.grid(row=5, column=0, padx=5, pady=5, sticky="W")

        ''' Class options frame '''
        self.class_frame = tk.LabelFrame(self.options_frame, text="Class Options")
        self.class_frame.grid(row=1, column=0, padx=10, pady=10, sticky="news")

        self.person_var = tk.StringVar(value="Off")
        self.person_option = tk.Checkbutton(self.class_frame, text="Person", variable=self.person_var, onvalue="On", offvalue="Off")
        self.person_option.grid(row=0, column=0, padx=5, pady=5, sticky="W")

        self.dog_var = tk.StringVar(value="Off")
        self.dog_option = tk.Checkbutton(self.class_frame, text="Dog", variable=self.dog_var, onvalue="On", offvalue="Off")
        self.dog_option.grid(row=0, column=1, padx=5, pady=5, sticky="W")

        self.cat_var = tk.StringVar(value="Off")
        self.cat_option = tk.Checkbutton(self.class_frame, text="Cat", variable=self.cat_var, onvalue="On", offvalue="Off")
        self.cat_option.grid(row=1, column=0, padx=5, pady=5, sticky="W")

        self.horse_var = tk.StringVar(value="Off")
        self.horse_option = tk.Checkbutton(self.class_frame, text="Horse", variable=self.horse_var, onvalue="On", offvalue="Off")
        self.horse_option.grid(row=1, column=1, padx=5, pady=5, sticky="W")

        self.cow_var = tk.StringVar(value="Off")
        self.cow_option = tk.Checkbutton(self.class_frame, text="Cow", variable=self.cow_var, onvalue="On", offvalue="Off")
        self.cow_option.grid(row=2, column=0, padx=5, pady=5, sticky="W")

        self.bird_var = tk.StringVar(value="Off")
        self.bird_option = tk.Checkbutton(self.class_frame, text="Bird", variable=self.bird_var, onvalue="On", offvalue="Off")
        self.bird_option.grid(row=2, column=1, padx=5, pady=5, sticky="W")

        self.car_var = tk.StringVar(value="Off")
        self.car_option = tk.Checkbutton(self.class_frame, text="Car", variable=self.car_var, onvalue="On", offvalue="Off")
        self.car_option.grid(row=3, column=0, padx=5, pady=5, sticky="W")

        self.bus_var = tk.StringVar(value="Off")
        self.bus_option = tk.Checkbutton(self.class_frame, text="Bus", variable=self.bus_var, onvalue="On", offvalue="Off")
        self.bus_option.grid(row=3, column=1, padx=5, pady=5, sticky="W")

        self.truck_var = tk.StringVar(value="Off")
        self.truck_option = tk.Checkbutton(self.class_frame, text="Truck", variable=self.truck_var, onvalue="On", offvalue="Off")
        self.truck_option.grid(row=4, column=0, padx=5, pady=5, sticky="W")

        self.motorcycle_var = tk.StringVar(value="Off")
        self.motorcycle_option = tk.Checkbutton(self.class_frame, text="Motorcycle", variable=self.motorcycle_var, onvalue="On", offvalue="Off")
        self.motorcycle_option.grid(row=4, column=1, padx=5, pady=5, sticky="W")

        self.mouse_var = tk.StringVar(value="Off")
        self.mouse_option = tk.Checkbutton(self.class_frame, text="Mouse", variable=self.mouse_var, onvalue="On", offvalue="Off")
        self.mouse_option.grid(row=5, column=0, padx=5, pady=5, sticky="W")

        self.cellphone_var = tk.StringVar(value="Off")
        self.cellphone_option = tk.Checkbutton(self.class_frame, text="Cell Phone", variable=self.cellphone_var, onvalue="On", offvalue="Off")
        self.cellphone_option.grid(row=5, column=1, padx=5, pady=5, sticky="W")

        self.keyboard_var = tk.StringVar(value="Off")
        self.keyboard_option = tk.Checkbutton(self.class_frame, text="Keyboard", variable=self.keyboard_var, onvalue="On", offvalue="Off")
        self.keyboard_option.grid(row=6, column=0, padx=5, pady=5, sticky="W")

        ''' buttons frame: Exit button '''
        self.btns_frame = tk.LabelFrame(self.options_frame, text="Buttons")
        self.btns_frame.grid(row=2, column=0, padx=10, pady=10, sticky="news")

        self.exit_btn = tk.Button(self.btns_frame, text="EXIT", padx=10, pady=10, width=22, command=self.close_window)
        self.exit_btn.grid(row=0, column=1, padx=5, pady=5, sticky="news")

        self.detection_model = YOLO('models/yolov8n.pt') # detection model shares with tracking model
        self.classification_model = YOLO('models/yolov8s-cls.pt')
        # self.segmentation_model = YOLO('models/yolov8s-seg.pt')
        self.pose_model = YOLO('models/yolov8n-pose.pt')

        self.video = cv2.VideoCapture(0);

        self.display_webcam()

    ''' take web cam image --> manipulated by machine models --> convert image format --> display on canvas '''
    def display_webcam(self):
        ret, frame = self.video.read()

        ''' detection mode with selected classes '''
        if(self.detection_var.get() == 'On'):
            self.set_selected_classes()
            # print(self.selected_classes)
            # frame = ym.execute_detection_model(self.detection_model, frame)
            frame = ym.execute_detection_model(self.detection_model, frame, self.selected_classes)

        ''' Classification and segmentation mode may be implemented in the future'''
        # if(self.classification_var.get() == 'On'):
        #     frame = ym.execute_classification_model(self.classification_model, frame)
        #     results = self.classification_model(frame)
        #
        # if(self.segmentation_var.get() == 'On'):
        #     frame = ym.execute_segmentation_model(self.segmentation_model, frame)
        ''' ====================================================================== '''

        ''' tracking mode with selected classes '''
        if(self.tracking_var.get() == 'On'):
            self.set_selected_classes()
            frame = ym.execute_tracking_mode(self.detection_model, frame, self.selected_classes)

        ''' pose estimation mode '''
        if(self.pose_var.get() == 'On'):
            frame = ym.execute_pose_mode(self.pose_model, frame)

        if ret:
            self.curr_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            self.photo = ImageTk.PhotoImage(self.curr_image)
            self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)
            # self.canvas.create_image(0, 0, image=frame, anchor=tk.NW)
            self.window.after(10, self.display_webcam)

    ''' set selected classes option in check-boxes on class option frame '''
    def set_selected_classes(self):
        self.selected_classes = []
        if(self.person_var.get() == "On"): self.selected_classes.append(0) # person class -> 0
        if(self.dog_var.get() == "On"): self.selected_classes.append(16) # dog class -> 16
        if(self.cat_var.get() == "On"): self.selected_classes.append(15) # cat class -> 15
        if(self.horse_var.get() == "On"): self.selected_classes.append(17) # person class -> 17
        if(self.cow_var.get() == "On"): self.selected_classes.append(19) # cow class -> 19
        if(self.bird_var.get() == "On"): self.selected_classes.append(14) # bird class -> 14
        if(self.car_var.get() == "On"): self.selected_classes.append(2) # car class -> 2
        if(self.bus_var.get() == "On"): self.selected_classes.append(5) # bus class -> 5
        if(self.truck_var.get() == "On"): self.selected_classes.append(7) # truck class -> 7
        if(self.motorcycle_var.get() == "On"): self.selected_classes.append(3) # motorcycle class -> 3
        if(self.mouse_var.get() == "On"): self.selected_classes.append(64) # mouse class -> 64
        if(self.cellphone_var.get() == "On"): self.selected_classes.append(67) # celiphone class -> 67
        if(self.keyboard_var.get() == "On"): self.selected_classes.append(66) # keyboard class -> 66
        # print(self.selected_classes)

    ''' execute EXIT button '''
    def close_window(self):
        self.window.destroy()

if __name__ == '__main__':
    root = tk.Tk()

    main_window = MainWindow(root)

    root.mainloop()