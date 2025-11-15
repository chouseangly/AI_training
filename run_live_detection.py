import cv2
import numpy as np
import mediapipe as mp
import joblib
import warnings


# --- SUPPRESS WARNINGS ---
warnings.filterwarnings('ignore', category=UserWarning, module='google.protobuf')

# --- LOAD THE TRAINED BEHAVIOR MODEL ---
try:
    model = joblib.load('drowning_model.pkl')
except FileNotFoundError:
    print("ERROR: Model file 'drowning_model.pkl' not found.")
    print("Please run the training script first to generate 'drowning_model.pkl'.")
    exit()

# --- INITIALIZE MEDIAPIPE POSE ---
mp_pose = mp.solutions.pose
pose_model = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# --- CONSTANTS ---
BEHAVIORS = ['drowning', 'normal', 'safe']
KEYPOINT_DIMENSION = 33 * 3  # 33 keypoints (x, y, z)


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

    # --- RUN POSE DETECTION ---
    pose_features, landmarks = get_pose_features(frame)

    if pose_features is not None and len(pose_features) == KEYPOINT_DIMENSION:
        features_for_prediction = pose_features.reshape(1, -1)

        # Safe prediction
        prediction_index = model.predict(features_for_prediction)[0]
        prediction_probabilities = model.predict_proba(features_for_prediction)[0]

        # Ensure index is within bounds
        if prediction_index >= len(BEHAVIORS):
            prediction_index = np.argmax(prediction_probabilities)

        current_behavior = BEHAVIORS[prediction_index]
        confidence = prediction_probabilities[prediction_index] * 100

        # Draw a GREEN box around the person (full frame) and behavior label
        cv2.putText(frame, f"{current_behavior} ({confidence:.1f}%)", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        # Optional: draw landmarks
        mp.solutions.drawing_utils.draw_landmarks(frame, landmarks, mp_pose.POSE_CONNECTIONS)

    # Show the frame
    cv2.imshow('Student Behavior Detection', frame)

    # Exit on 'q'
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

# --- CLEANUP ---
cap.release()
cv2.destroyAllWindows()
