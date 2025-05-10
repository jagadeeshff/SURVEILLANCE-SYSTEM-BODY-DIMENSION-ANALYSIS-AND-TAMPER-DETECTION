import cv2
import numpy as np
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
from email.mime.text import MIMEText
import time
import threading
import os
import csv
from ultralytics import YOLO

# Suppress YOLOv8 verbose logs
os.environ["YOLO_VERBOSE"] = "0"

# Load YOLOv8 model for weapon detection
weapon_model = YOLO("best4.pt")

# Known width and height in pixels and their corresponding real-world dimensions in cm
KNOWN_WIDTH_PIXELS = 40
KNOWN_WIDTH_CM = 60
KNOWN_HEIGHT_PIXELS = 80
KNOWN_HEIGHT_CM = 170

# Scale factors
SCALE_FACTOR_WIDTH_CM_PER_PIXEL = KNOWN_WIDTH_CM / KNOWN_WIDTH_PIXELS
SCALE_FACTOR_HEIGHT_CM_PER_PIXEL = KNOWN_HEIGHT_CM / KNOWN_HEIGHT_PIXELS

# Load pre-trained models for gender and age detection
AGE_MODEL = "age_net.caffemodel"
AGE_PROTO = "age_deploy.prototxt"
GENDER_MODEL = "gender_net.caffemodel"
GENDER_PROTO = "gender_deploy.prototxt"

# Load the models
age_net = cv2.dnn.readNetFromCaffe(AGE_PROTO, AGE_MODEL)
gender_net = cv2.dnn.readNetFromCaffe(GENDER_PROTO, GENDER_MODEL)

# Define labels for age and gender
AGE_CLASSES = ["(0-2)", "(4-6)", "(8-12)", "(15-20)", "(21-25)", "(26-32)", "(33-45)", "(46-55)", "(56-100)"]
GENDER_CLASSES = ["Male", "Female"]

# Load OpenCV's face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Initialize MediaPipe Pose model
import mediapipe as mp
mp_pose = mp.solution. s.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Open webcam
cap = cv2.VideoCapture(0)

# Ensure the captured images directory exists
capture_dir = "captured_images"
os.makedirs(capture_dir, exist_ok=True)

# Logging setup (CSV file to store measurements)
log_file = 'measurements.csv'

# Write headers to CSV if file doesn't exist
if not os.path.exists(log_file):
    with open(log_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Timestamp', 'Width (cm)', 'Height (cm)', 'Weapon Detected', 'Gender', 'Age'])

# Email sending function
def send_email(image_path, width, height, gender, age):
    sender_email = "topjrealworld@gmail.com"
    receiver_email = "jagadeeshm2612@gmail.com"
    password = "twyapmxcoqaugebz"
    
    # Prepare the email
    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = receiver_email
    msg['Subject'] = "Weapon Detected Alert"

    # Attach the image
    with open(image_path, 'rb') as f:
        img = MIMEImage(f.read())
        msg.attach(img)

    # Email body with added information (height, width, gender, and age)
    body = MIMEText(f"Weapon detected!\nHeight: {height} cm\nWidth: {width} cm\nGender: {gender}\nAge: {age}\nPlease find the attached image.")
    msg.attach(body)

    try:
        # Establish connection to the SMTP server
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(sender_email, password)
        server.sendmail(sender_email, receiver_email, msg.as_string())
        server.quit()
        print("[INFO] Email sent successfully!")
    except Exception as e:
        print(f"[ERROR] Failed to send email: {str(e)}")

# Function to send email asynchronously
def send_email_async(image_path, width, height, gender, age):
    thread = threading.Thread(target=send_email, args=(image_path, width, height, gender, age))
    thread.start()

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

    # Run YOLOv8 detection for weapon detection
    weapon_detected = False
    detections = weapon_model(frame, verbose=False)

    for detection in detections:
        for box in detection.boxes:
            confidence = float(box.conf[0])

            # If confidence is above 0.50, mark as weapon
            if confidence > 0.50:
                weapon_detected = True
                # Draw bounding box and label
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                label = f"Weapon {confidence:.2f}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Age and Gender Prediction
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        face = frame[y:y + h, x:x + w]
        blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), (78.426, 87.768, 114.895), swapRB=False)

        # Predict Gender
        gender_net.setInput(blob)
        gender_pred = gender_net.forward()
        gender = GENDER_CLASSES[gender_pred[0].argmax()]

        # Predict Age
        age_net.setInput(blob)
        age_pred = age_net.forward()
        age = AGE_CLASSES[age_pred[0].argmax()]

        # Draw bounding box and label
        label = f"{gender}, Age: {age}"
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Capture and log data if a full body is detected AND a weapon is detected
    if result.pose_landmarks and is_full_body_detected(result.pose_landmarks.landmark) and weapon_detected:
        timestamp = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
        image_filename = f"{capture_dir}/alert_{timestamp}.jpg"
        cv2.imwrite(image_filename, frame)  # Save image

        with open(log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([timestamp, width_in_cm, height_in_cm, "Yes", gender, age])

        print(f"[ALERT] Weapon detected! Image saved: {image_filename}")

        # Send email asynchronously with all info
        send_email_async(image_filename, width_in_cm, height_in_cm, gender, age)

    # Display the image with detections
    cv2.imshow("Weapon & Body Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to exit
        break

cap.release()
cv2.destroyAllWindows()
