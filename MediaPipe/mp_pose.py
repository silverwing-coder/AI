import time
import cv2
import mediapipe as mp

# Short aliases for the newer MediaPipe Tasks API
BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# Global variable to store the asynchronous results
latest_result = None

def print_result(result, output_image, timestamp_ms):
    """Callback function that receives the detection results."""
    global latest_result
    latest_result = result

# Configure Pose Landmarker Options
options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='pose_landmarker_full.task'),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=print_result,
    output_segmentation_masks=False  # Set to True if you want a background mask
)

# Initialize OpenCV webcam stream
cap = cv2.VideoCapture(0)

# Initialize landmarker using a context manager
with PoseLandmarker.create_from_options(options) as landmarker:
    prev_time = 0
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        # Flip horizontally for selfie view, convert BGR to RGB
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Convert OpenCV frame to MediaPipe Image object
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        # Generate a monotonically increasing timestamp in milliseconds
        frame_timestamp_ms = int(time.time() * 1000)
        
        # Send frame to the landmarker asynchronously
        landmarker.detect_async(mp_image, frame_timestamp_ms)

        # Draw landmarks if they are available from the callback
        if latest_result is not None and latest_result.pose_landmarks:
            for pose_landmarks in latest_result.pose_landmarks:
                # Loop through all 33 detected landmarks
                for landmark in pose_landmarks:
                    # Convert normalized coordinates back to pixel coordinates
                    h, w, _ = frame.shape
                    cx, cy = int(landmark.x * w), int(landmark.y * h)
                    
                    # Draw a small circle on every keypoint
                    if landmark.visibility > 0.5:  # Only draw visible points
                        cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)

        current_time = time.time()
        fps = int(1 / (current_time - prev_time))
        prev_time = time.time()
                  

        cv2.putText(frame, f'FPS: {int(fps)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Display the output
        cv2.imshow('MediaPipe Tasks - Pose Landmarker', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()