import cv2
import numpy as np
import csv
import time
import os
import mediapipe as mp
import smtplib
from email.message import EmailMessage

# -------------------- EMAIL ALERT FUNCTION --------------------

def send_email_alert(subject, body):
    EMAIL_ADDRESS = 'topjrealworld@gmail.com'
    EMAIL_PASSWORD = 'twyapmxcoqaugebz'  # Use App Password if using Gmail
    ALERT_RECEIVER = 'jagadeeshm2612@gmail.com'

    msg = EmailMessage()
    msg['Subject'] = subject
    msg['From'] = EMAIL_ADDRESS
    msg['To'] = ALERT_RECEIVER
    msg.set_content(body)

    try:
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
            smtp.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
            smtp.send_message(msg)
        print("[EMAIL ALERT] Sent successfully.")
    except Exception as e:
        print(f"[EMAIL ERROR] Failed to send alert: {e}")

# -------------------- SETUP --------------------

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

cap = cv2.VideoCapture("rtsp://admin:Jaga777@@192.168.1.108:554/cam/realmonitor?channel=1&subtype=0")

capture_dir = "captured_images"
os.makedirs(capture_dir, exist_ok=True)

log_file = 'measurements.csv'

if not os.path.exists(log_file):
    with open(log_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Timestamp', 'Width (cm)', 'Height (cm)'])

# Scale factors (adjust based on your setup)
KNOWN_WIDTH_PIXELS = 40
KNOWN_WIDTH_CM = 60
KNOWN_HEIGHT_PIXELS = 80
KNOWN_HEIGHT_CM = 170
SCALE_FACTOR_WIDTH_CM_PER_PIXEL = KNOWN_WIDTH_CM / KNOWN_WIDTH_PIXELS
SCALE_FACTOR_HEIGHT_CM_PER_PIXEL = KNOWN_HEIGHT_CM / KNOWN_HEIGHT_PIXELS

def is_full_body_detected(landmarks):
    required_landmarks = [
        mp_pose.PoseLandmark.NOSE, 
        mp_pose.PoseLandmark.LEFT_ANKLE, 
        mp_pose.PoseLandmark.RIGHT_ANKLE, 
        mp_pose.PoseLandmark.LEFT_KNEE, 
        mp_pose.PoseLandmark.RIGHT_KNEE
    ]
    return all(landmarks[lm].visibility > 0.5 for lm in required_landmarks)

def is_black_screen(frame, threshold=10):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return np.mean(gray) < threshold

last_capture_time = 0
capture_interval = 10
last_black_screen_alert = 0
black_screen_alert_interval = 120  # alert once every 5 minutes if continuous

# -------------------- MAIN LOOP --------------------

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)

    # -------------------- BLACK SCREEN DETECTION --------------------
    if is_black_screen(frame):
        current_time = time.time()
        if current_time - last_black_screen_alert > black_screen_alert_interval:
            last_black_screen_alert = current_time
            send_email_alert("⚠️ Black Screen Alert", "The camera feed appears to be black. Please check the setup.")
        cv2.imshow("Frame", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = pose.process(rgb_frame)

    if result.pose_landmarks:
        mp.solutions.drawing_utils.draw_landmarks(frame, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        left_shoulder = result.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = result.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        nose = result.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE]
        left_ankle = result.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE]

        width_pixels = np.linalg.norm([left_shoulder.x - right_shoulder.x, left_shoulder.y - right_shoulder.y])
        height_pixels = np.linalg.norm([nose.x - left_ankle.x, nose.y - left_ankle.y])

        width_cm = int(width_pixels * SCALE_FACTOR_WIDTH_CM_PER_PIXEL * 100)
        height_cm = int(height_pixels * SCALE_FACTOR_HEIGHT_CM_PER_PIXEL * 100 + 10)

        cv2.putText(frame, f"Width: {width_cm} cm", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Height: {height_cm} cm", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        current_time = time.time()
        if is_full_body_detected(result.pose_landmarks.landmark) and (current_time - last_capture_time >= capture_interval):
            last_capture_time = current_time
            timestamp = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
            image_filename = f"{capture_dir}/alert_{timestamp}.jpg"
            cv2.imwrite(image_filename, frame)

            with open(log_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([timestamp, width_cm, height_cm])

            print(f"[ALERT] Full body detected! Image saved: {image_filename}")

    cv2.imshow("Frame", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
