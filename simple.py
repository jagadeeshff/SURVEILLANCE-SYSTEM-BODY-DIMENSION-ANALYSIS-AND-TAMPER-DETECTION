import cv2
import numpy as np
import csv
import time
import os
import mediapipe as mp

# Load MediaPipe Pose model
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Open webcam
cap = cv2.VideoCapture("rtsp://admin:Jaga777@@192.168.1.108:554/cam/realmonitor?channel=1&subtype=0")
# Ensure the captured images directory exists
capture_dir = "captured_images"
os.makedirs(capture_dir, exist_ok=True)

# Logging setup (CSV file to store measurements)
log_file = 'measurements.csv'

# Write headers to CSV if file doesn't exist
if not os.path.exists(log_file):
    with open(log_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Timestamp', 'Width (cm)', 'Height (cm)'])

# Known width and height in pixels and their corresponding real-world dimensions in cm
KNOWN_WIDTH_PIXELS = 40
KNOWN_WIDTH_CM = 60
KNOWN_HEIGHT_PIXELS = 80
KNOWN_HEIGHT_CM = 170

# Scale factors
SCALE_FACTOR_WIDTH_CM_PER_PIXEL = KNOWN_WIDTH_CM / KNOWN_WIDTH_PIXELS
SCALE_FACTOR_HEIGHT_CM_PER_PIXEL = KNOWN_HEIGHT_CM / KNOWN_HEIGHT_PIXELS

# Function to check if full body is detected
def is_full_body_detected(landmarks):
    required_landmarks = [
        mp_pose.PoseLandmark.NOSE, 
        mp_pose.PoseLandmark.LEFT_ANKLE, 
        mp_pose.PoseLandmark.RIGHT_ANKLE, 
        mp_pose.PoseLandmark.LEFT_KNEE, 
        mp_pose.PoseLandmark.RIGHT_KNEE
    ]
    return all(landmarks[lm].visibility > 0.5 for lm in required_landmarks)

# Interval configuration
last_capture_time = 0
capture_interval = 10  # seconds

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the frame horizontally
    frame = cv2.flip(frame, 1)

    # Convert the BGR image to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Perform pose detection
    result = pose.process(rgb_frame)

    if result.pose_landmarks:
        # Draw landmarks on the frame
        mp.solutions.drawing_utils.draw_landmarks(frame, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Extract specific body parts for width/height calculation
        left_shoulder = result.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = result.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        nose = result.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE]
        left_ankle = result.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE]
        right_ankle = result.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE]

        # Calculate width and height in pixels
        width_in_pixels = ((right_shoulder.x - left_shoulder.x) ** 2 + (right_shoulder.y - left_shoulder.y) ** 2) ** 0.5
        height_in_pixels = ((nose.x - left_ankle.x) ** 2 + (nose.y - left_ankle.y) ** 2) ** 0.5

        # Convert width and height to centimeters
        width_in_cm = int(width_in_pixels * SCALE_FACTOR_WIDTH_CM_PER_PIXEL * 100)
        height_in_cm = int(height_in_pixels * SCALE_FACTOR_HEIGHT_CM_PER_PIXEL * 100 + 10)

        # Display width and height on frame
        cv2.putText(frame, f"Width: {width_in_cm} cm", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, f"Height: {height_in_cm} cm", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # Capture and log data at intervals
        current_time = time.time()
        if is_full_body_detected(result.pose_landmarks.landmark) and (current_time - last_capture_time >= capture_interval):
            last_capture_time = current_time
            timestamp = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
            image_filename = f"{capture_dir}/alert_{timestamp}.jpg"
            cv2.imwrite(image_filename, frame)  # Save image

            with open(log_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([timestamp, width_in_cm, height_in_cm])

            print(f"[ALERT] Full body detected! Image saved: {image_filename}")

    # Display the frame
    cv2.imshow("Frame", frame)

    # Exit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
