import cv2
import numpy as np
import os
import random
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
# Import MediaPipe for robust pose estimation
import mediapipe as mp

# Initialize MediaPipe Pose Model
mp_pose = mp.solutions.pose
# Use the Lite model for faster inference if necessary
pose_model = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# --- CONSTANTS AND CONFIGURATION ---
DATA_DIR = 'classroom_videos/labeled_data'
BEHAVIORS = ['Attentive', 'Inattentive', 'Talking']
FRAME_SKIP = 5  # Process every 5th frame to speed up training
KEYPOINT_DIMENSION = 33 * 3  # 33 keypoints (x, y, z) = 99 features per frame


# --- STAGE 1 & 2: FEATURE EXTRACTION (The Core Function) ---

def get_pose_features(frame):
    """
    Uses the MediaPipe Pose model to extract a feature vector (pose keypoints).

    Returns a flattened numpy array of (x, y, z) coordinates for all 33 body joints.
    Returns None if no human is detected.
    """
    # Convert the BGR image to RGB
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False  # Make image non-writeable for better performance

    # Process the image and get the pose results
    results = pose_model.process(image)

    image.flags.writeable = True  # Make image writeable again

    features = []

    if results.pose_landmarks:
        # We iterate through all 33 landmarks
        for landmark in results.pose_landmarks.landmark:
            # We are extracting x, y, and z coordinates
            features.append(landmark.x)
            features.append(landmark.y)
            features.append(landmark.z)

        return np.array(features)

    return None


def extract_features_from_video(video_path, behavior_index):
    """Loads video, extracts pose features every N frames, and labels them."""

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return [], []

    features_list = []
    labels_list = []
    frame_count = 0

    print(f"-> Processing video: {os.path.basename(video_path)}...")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % FRAME_SKIP == 0:
            pose_features = get_pose_features(frame)

            if pose_features is not None and len(pose_features) == KEYPOINT_DIMENSION:
                features_list.append(pose_features)
                labels_list.append(behavior_index)  # Use the numeric index as the label

        frame_count += 1

    cap.release()
    print(f"   Extracted {len(features_list)} frames (features) successfully.")
    return features_list, labels_list


def load_and_prepare_data():
    """Iterates through the data directory and extracts features from all videos."""

    all_features = []
    all_labels = []

    if not os.path.isdir(DATA_DIR):
        print(f"ERROR: Data directory '{DATA_DIR}' not found. Please create it and add videos.")
        return np.array([]), np.array([])

    # Iterate through all defined behaviors (subdirectories)
    for index, behavior in enumerate(BEHAVIORS):
        behavior_path = os.path.join(DATA_DIR, behavior)
        if os.path.isdir(behavior_path):

            # Find all video files (.mp4, .mov, etc.)
            video_files = [f for f in os.listdir(behavior_path) if f.lower().endswith(('.mp4', '.mov', '.avi'))]

            print(f"\nFound {len(video_files)} videos for behavior: {behavior}")

            for video_name in video_files:
                video_path = os.path.join(behavior_path, video_name)

                # Execute feature extraction for the video
                features, labels = extract_features_from_video(video_path, index)

                all_features.extend(features)
                all_labels.extend(labels)

    # Convert lists to numpy arrays
    X = np.array(all_features)
    y = np.array(all_labels)

    print(f"\n--- Data Preparation Complete ---")
    print(f"Total Feature Vectors (Frames): {X.shape[0]}")
    print(f"Feature Dimension: {X.shape[1] if X.shape[0] > 0 else 0}")

    return X, y


# --- STAGE 3 & 4: MODEL TRAINING AND EVALUATION ---

def train_behavior_classifier(X, y):
    """Trains a Random Forest classifier and prints the test accuracy."""

    # 1. Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    print(f"\nTraining Model with {X_train.shape[0]} samples...")

    # 2. Initialize and Train the Model
    # Random Forest is robust and works well on structured feature data like this
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    # 3. Evaluation
    accuracy = model.score(X_test, y_test)
    print("Model Training Complete.")
    print(f"\n--- Model Performance ---")
    print(f"Test Accuracy: {accuracy * 100:.2f}%")
    print("-------------------------")
    print("Model trained and ready for deployment!")

    return model


if __name__ == '__main__':
    print("--- Starting Student Behavior AI Model Workflow ---")

    # 1. LOAD AND PREPARE DATA (Feature Extraction)
    X_features, y_labels = load_and_prepare_data()

    if X_features.shape[0] == 0:
        print("\n[STOPPED] No data was loaded. Please ensure your videos are in the correct directory structure:")
        print(f"  {DATA_DIR}/Attentive/")
        print(f"  {DATA_DIR}/Inattentive/")
        print("Run `pip install ...` if you haven't already.")
    else:
        # 2. TRAIN CLASSIFIER MODEL
        trained_model = train_behavior_classifier(X_features, y_labels)

        # 3. Deployment Prep (Save the model for later use)
        # To use this model later, you would save it using joblib or pickle:
        # import joblib
        # joblib.dump(trained_model, 'student_behavior_model.pkl')
        # print("\nModel saved as 'student_behavior_model.pkl'.")

        # NOTE: After training, you would use this model to predict behavior
        # on new, unseen, live video feeds.
