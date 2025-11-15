import cv2
import numpy as np
import mediapipe as mp
import joblib
import warnings
import os

# --- NEW: Import MediaPipe Tasks ---
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# --- SUPPRESS WARNINGS ---
warnings.filterwarnings('ignore', category=UserWarning, module='google.protobuf')

# --- LOAD THE TRAINED BEHAVIOR MODEL ---
try:
    model = joblib.load('student_behavior_model.pkl')
except FileNotFoundError:
    print("ERROR: Model file 'student_behavior_model.pkl' not found.")
    print("Please run the main.py script first to train and save the model.")
    exit()

# --- INITIALIZE MEDIAPIPE ---
# 1. Object Detector
model_path = 'efficientdet_lite0.tflite'

if not os.path.exists(model_path):
    print(f"ERROR: Model file '{model_path}' not found.")
    print(
        "Please download it from: https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite0/int8/1/efficientdet_lite0.tflite")
    print("And place it in the same folder as this script.")
    exit()

# Create options for the Object Detector
base_options = python.BaseOptions(model_asset_path=model_path)

# --- THIS IS THE FIX ---
# We now tell the detector to look for people, laptops, and cell phones
options = vision.ObjectDetectorOptions(
    base_options=base_options,
    score_threshold=0.5,
    category_allowlist=['person', 'laptop', 'cell phone']  # <-- UPDATED LIST
)

try:
    object_detector = vision.ObjectDetector.create_from_options(options)
except Exception as e:
    print(f"ERROR: Could not load the Object Detector model.")
    print(f"Full error: {e}")
    exit()

# 2. Pose Model
mp_pose = mp.solutions.pose
pose_model = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# --- CONSTANTS ---
BEHAVIORS = ['Attentive', 'Inattentive', 'Talking', 'Sleeping']
KEYPOINT_DIMENSION = 33 * 3  # 99 features


def get_pose_features(frame):
    """Extracts pose features from a single frame."""
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = pose_model.process(image)
    image.flags.writeable = True

    features = []
    if results.pose_landmarks:
        for landmark in results.pose_landmarks.landmark:
            features.extend([landmark.x, landmark.y, landmark.z])
        return np.array(features), results.pose_landmarks
    return None, None


# --- MAIN WEBCAM LOOP ---
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 1. Run Object Detection
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
    detection_result = object_detector.detect(mp_image)

    if detection_result.detections:
        # Loop over every object detected
        for detection in detection_result.detections:

            # Get the category name and bounding box
            category = detection.categories[0]
            category_name = category.category_name

            box = detection.bounding_box
            h, w, _ = frame.shape
            x_min = int(box.origin_x)
            y_min = int(box.origin_y)
            x_max = int(box.origin_x + box.width)
            y_max = int(box.origin_y + box.height)

            # Ensure coordinates are valid
            x_min = max(0, x_min)
            y_min = max(0, y_min)
            x_max = min(w, x_max)
            y_max = min(h, y_max)

            # --- NEW LOGIC: Check WHAT we detected ---

            if category_name == 'person':
                # --- This is your original logic for people ---
                person_crop = frame[y_min:y_max, x_min:x_max]

                if person_crop.shape[0] > 0 and person_crop.shape[1] > 0:
                    pose_features, _ = get_pose_features(person_crop)

                    if pose_features is not None and len(pose_features) == KEYPOINT_DIMENSION:
                        features_for_prediction = pose_features.reshape(1, -1)
                        prediction_index = model.predict(features_for_prediction)[0]
                        prediction_probabilities = model.predict_proba(features_for_prediction)[0]

                        current_behavior = BEHAVIORS[prediction_index]
                        confidence = prediction_probabilities[prediction_index] * 100

                        # Draw GREEN box for people + behavior
                        label_text = f"{current_behavior} ({confidence:.1f}%)"
                        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                        cv2.rectangle(frame, (x_min, y_min - 25), (x_max, y_min), (0, 255, 0), -1)
                        cv2.putText(frame, label_text, (x_min + 5, y_min - 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)

            elif category_name == 'laptop' or category_name == 'cell phone':
                # --- This is the new logic for objects ---
                # We just draw a box and a label, no behavior analysis

                # Draw BLUE box for objects
                label_text = category_name.upper()
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)
                cv2.rectangle(frame, (x_min, y_min - 25), (x_max, y_min), (255, 0, 0), -1)
                cv2.putText(frame, label_text, (x_min + 5, y_min - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

    # Show the final frame
    cv2.imshow('Student Behavior Detection (Multi-Class)', frame)

    # Exit loop if 'q' is pressed
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

# --- CLEANUP ---
cap.release()
cv2.destroyAllWindows()