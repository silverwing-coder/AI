import cv2
import tkinter as tk
from PIL import Image, ImageTk

import ModuleObjectDection as mod
import ModuleHands as mh
import ModuleFaceDetect as fd
import ModuleFaceMesh as fm
import ModulePoseEstimation as pe

class MainMediaPipe:

    def __init__(self, window):
        self.window = window

        self.canvas_frame = tk.LabelFrame(self.window, text="CANVAS")
        self.canvas_frame.grid(row=0, column=0, padx=10, pady=10)

        self.canvas = tk.Canvas(self.canvas_frame, width=640, height=480)
        self.canvas.pack()

        self.options_frame = tk.LabelFrame(self.window, text="OPTIONS")
        self.options_frame.grid(row=0, column=1, padx=10, pady=10, sticky="news")

        self.objectdetect_var = tk.BooleanVar(value=False)
        self.objectdetect_option = tk.Checkbutton(self.options_frame, text="Object Detection Mode",
                                                  variable=self.objectdetect_var, onvalue=True, offvalue=False)
        self.objectdetect_option.grid(row=0, column=0, padx=5, pady=5, sticky="W")

        self.hands_var = tk.BooleanVar(value=False)
        self.hands_option = tk.Checkbutton(self.options_frame, text="Hand Landmarks Mode",
                                           variable=self.hands_var, onvalue=True,offvalue=False)
        self.hands_option.grid(row=1, column=0, padx=5, pady=5, sticky="W")

        self.facedetect_var = tk.BooleanVar(value=False)
        self.facedetect_option = tk.Checkbutton(self.options_frame, text="Face Detection Mode",
                                                variabl=self.facedetect_var, onvalue=True, offvalue=False)
        self.facedetect_option.grid(row=2, column=0, padx=5, pady=5, sticky="W")

        self.facemesh_var = tk.BooleanVar(value=False)
        self.facemesh_option = tk.Checkbutton(self.options_frame, text="Face Landmark Mode",
                                              variable=self.facemesh_var, onvalue=True, offvalue=False)
        self.facemesh_option.grid(row=3, column=0, padx=5, pady=5, sticky="W")

        self.pose_var = tk.BooleanVar(value=False)
        self.pose_option = tk.Checkbutton(self.options_frame, text="Pose Landmark Mode",
                                          variable=self.pose_var, onvalue=True, offvalue=False)
        self.pose_option.grid(row=4, column=0, padx=5, pady=5, sticky="W")

        self.buttons_frame = tk.LabelFrame(self.options_frame, text="Buttons")
        self.buttons_frame.grid(row=5, column=0, padx=5, pady=5)

        self.exit_buton = tk.Button(self.buttons_frame, text="EXIT", padx=10, pady=10, width=20, command=self.close_window)
        self.exit_buton.grid(row=0, column=0, padx=5, pady=5, sticky="news")

        self.video = cv2.VideoCapture(0)
        self.display_webcam()


    def display_webcam(self):
        ret, image = self.video.read()
        image = cv2.flip(image, 1)

        if(self.objectdetect_var.get()):
            results = mod.get_object_detection_results(image);
            image = mod.get_annotated_image_objects(image, results)
            # print(self.objectdetect_var.get())

        if(self.hands_var.get()):
            results = mh.get_hands_detection_results(image)
            image = mh.get_annotated_image_hands(image, results)

        if(self.facedetect_var.get()):
            results = fd.get_face_detected_results(image)
            image = fd.get_annotated_image_faces(image, results)

        if(self.facemesh_var.get()):
            results = fm.get_facemesh_results(image)
            image =fm.get_annotated_image_facemesh(image, results)
            # print(results)

        if(self.pose_var.get()):
            results = pe.get_pose_estimation_results(image)
            image = pe.get_annotated_image_pose(image, results)
            # print(self.pose_var.get())

        if ret:
            # image = cv2.flip(image, 1)
            self.image_array = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            self.photo_image = ImageTk.PhotoImage(self.image_array)
            self.canvas.create_image(0, 0, image=self.photo_image, anchor=tk.NW)
            self.window.after(10, self.display_webcam)

    ''' execute EXIT button '''

    def close_window(self):
        self.window.destroy()

if __name__ == '__main__':

    window = tk.Tk()
    window.title("MediaPipe on Tkinter")

    application = MainMediaPipe(window)

    window.mainloop()