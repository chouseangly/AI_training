import cv2
import numpy as np
import mediapipe as mp
import joblib
import warnings

# --- Try to import TFLite runtime, fall back to full TensorFlow ---
try:
    from tflite_runtime.interpreter import Interpreter
except ImportError:
    print("tflite_runtime not found. Trying full TensorFlow...")
    try:
        from tensorflow.lite.python.interpreter import Interpreter
    except ImportError:
        print("ERROR: Failed to import TFLite Interpreter.")
        print("Please install TensorFlow or tflite-runtime: pip install tensorflow")
        exit()

# --- SUPPRESS WARNINGS ---
warnings.filterwarnings('ignore', category=UserWarning, module='google.protobuf')

# === STAGE 1: OBJECT DETECTION (from efficientdet_lite0.tflite) ===
OBJECT_MODEL_PATH = 'efficientdet_lite0.tflite'
LABEL_PATH = 'labelmap.txt'
CONFIDENCE_THRESHOLD = 0.5  # Only detect objects with 50% or higher confidence


def load_labels(path):
    """Loads the label map"""
    with open(path, 'r') as f:
        return {int(i): line.strip() for i, line in enumerate(f.readlines())}


# --- Load Object Detection Model ---
try:
    interpreter = Interpreter(model_path=OBJECT_MODEL_PATH)
    interpreter.allocate_tensors()
    print("Object detection model loaded.")
except ValueError:
    print(f"ERROR: Model file '{OBJECT_MODEL_PATH}' not found.")
    exit()

# Get model input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
INPUT_HEIGHT = input_details[0]['shape'][1]
INPUT_WIDTH = input_details[0]['shape'][2]

# Load labels
try:
    labels = load_labels(LABEL_PATH)
    print("Label map loaded.")
except FileNotFoundError:
    print(f"ERROR: Label file '{LABEL_PATH}' not found.")
    print("Please create this file and add the COCO class names.")
    exit()

# === STAGE 2: BEHAVIOR CLASSIFICATION (from drowning_model.pkl) ===
BEHAVIOR_MODEL_PATH = 'drowning_model.pkl'

# --- Load Behavior Model ---
try:
    behavior_model = joblib.load(BEHAVIOR_MODEL_PATH)
    print("Behavior classification model loaded.")
except FileNotFoundError:
    print(f"ERROR: Model file '{BEHAVIOR_MODEL_PATH}' not found.")
    exit()

# --- Initialize MediaPipe Pose ---
mp_pose = mp.solutions.pose
pose_model = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)
print("MediaPipe Pose initialized.")

# --- Behavior Constants ---
# !! Make sure this matches your trained model !!
BEHAVIORS = ['drowning', 'normal', 'safe']
KEYPOINT_DIMENSION = 33 * 3  # 33 keypoints (x, y, z)


def get_pose_features(frame):
    """Extracts pose features from a single frame."""
    # This function is copied from your original script
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = pose_model.process(image)
    image.flags.writeable = True

    features = []
    if results.pose_landmarks:
        for landmark in results.pose_landmarks.landmark:
            features.extend([landmark.x, landmark.y, landmark.z])
        return np.array(features)
    return None


# ===================================================================
# --- MAIN WEBCAM LOOP ---
# ===================================================================

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Get original frame dimensions
    original_height, original_width, _ = frame.shape

    # --- STAGE 1: Run Object Detection ---
    # Resize frame for object detection model
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_resized = cv2.resize(frame_rgb, (INPUT_WIDTH, INPUT_HEIGHT))
    input_data = np.expand_dims(frame_resized, axis=0)

    # Perform inference
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    # Get results
    boxes = interpreter.get_tensor(output_details[0]['index'])[0]
    classes = interpreter.get_tensor(output_details[1]['index'])[0]
    scores = interpreter.get_tensor(output_details[2]['index'])[0]

    # --- STAGE 2: Process Detections and Run Behavior Analysis ---
    for i in range(len(scores)):
        if scores[i] >= CONFIDENCE_THRESHOLD:
            # Get class name
            class_id = int(classes[i])
            class_name = labels.get(class_id, f'Class {class_id}')

            # Get bounding box coordinates and scale them to original frame size
            y_min, x_min, y_max, x_max = boxes[i]
            x_min = int(max(1, x_min * original_width))
            y_min = int(max(1, y_min * original_height))
            x_max = int(min(original_width, x_max * original_width))
            y_max = int(min(original_height, y_max * original_height))

            # Default color and label
            color = (10, 255, 0)  # Green
            label = f'{class_name}: {int(scores[i] * 100)}%'

            # --- IF OBJECT IS A PERSON, RUN BEHAVIOR ANALYSIS ---
            if class_name == 'person':
                # Crop the person from the *original* frame
                person_crop = frame[y_min:y_max, x_min:x_max]

                # Ensure crop is valid before processing
                if person_crop.shape[0] > 0 and person_crop.shape[1] > 0:

                    # Run pose detection on the crop
                    pose_features = get_pose_features(person_crop)

                    if pose_features is not None and len(pose_features) == KEYPOINT_DIMENSION:
                        # --- Run Behavior Prediction ---
                        features_for_prediction = pose_features.reshape(1, -1)
                        prediction_probabilities = behavior_model.predict_proba(features_for_prediction)[0]
                        prediction_index = np.argmax(prediction_probabilities)
                        confidence = prediction_probabilities[prediction_index] * 100

                        if prediction_index < len(BEHAVIORS):
                            current_behavior = BEHAVIORS[prediction_index]
                        else:
                            current_behavior = "Unknown"

                        # Update label with behavior
                        label = f"{current_behavior} ({confidence:.1f}%)"

                        # Change color based on behavior
                        if current_behavior == 'drowning':
                            color = (0, 0, 255)  # Red
                        else:
                            color = (0, 255, 0)  # Green

                    else:
                        # Pose not detected in crop, just label as person
                        label = "person (No Pose)"
                        color = (0, 255, 255)  # Yellow

            # --- Draw Box and Label for ALL objects ---
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 2)
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(frame, (x_min, y_min - h - 10), (x_min + w, y_min), color, -1)
            cv2.putText(frame, label, (x_min, y_min - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)

    # Show the frame
    cv2.imshow('Multi-Object Behavior Detection', frame)

    # Exit on 'q'
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

# --- CLEANUP ---
cap.release()
cv2.destroyAllWindows()