import cv2
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
# Import MediaPipe for robust pose estimation
import mediapipe as mp
# Import joblib to save the model
import joblib

# Initialize MediaPipe Pose Model
mp_pose = mp.solutions.pose
pose_model = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# --- CONSTANTS AND CONFIGURATION ---
DATA_DIR = 'classroom_images/labeled_data'  # Directory for images

# UPDATED: Added 'Sleeping' to the list
BEHAVIORS = ['Attentive', 'Inattentive', 'Talking', 'Sleeping']

# FRAME_SKIP is no longer needed
KEYPOINT_DIMENSION = 33 * 3  # 33 keypoints (x, y, z) = 99 features


# --- STAGE 1 & 2: FEATURE EXTRACTION (The Core Function) ---

def get_pose_features(frame):
    """
    Uses the MediaPipe Pose model to extract a feature vector (pose keypoints).
    This function works perfectly on a single image (frame).
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


def load_and_prepare_data():
    """
    MODIFIED FUNCTION:
    Iterates through the data directory, loads each IMAGE,
    extracts pose features, and labels them.
    """

    all_features = []
    all_labels = []

    if not os.path.isdir(DATA_DIR):
        print(f"ERROR: Data directory '{DATA_DIR}' not found. Please create it and add images.")
        return np.array([]), np.array([])

    # Define valid image extensions
    image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')

    # Iterate through all defined behaviors (subdirectories)
    for index, behavior in enumerate(BEHAVIORS):
        behavior_path = os.path.join(DATA_DIR, behavior)
        if os.path.isdir(behavior_path):

            # Find all image files
            image_files = [f for f in os.listdir(behavior_path) if f.lower().endswith(image_extensions)]

            print(f"\nFound {len(image_files)} images for behavior: {behavior}")

            for image_name in image_files:
                image_path = os.path.join(behavior_path, image_name)

                # Load the image
                frame = cv2.imread(image_path)

                if frame is None:
                    print(f"   Warning: Could not read image {image_name}. Skipping.")
                    continue

                # Execute feature extraction for the single image
                pose_features = get_pose_features(frame)

                if pose_features is not None and len(pose_features) == KEYPOINT_DIMENSION:
                    all_features.append(pose_features)
                    all_labels.append(index)  # Use the numeric index as the label
                else:
                    print(f"   Warning: No pose detected in {image_name}. Skipping.")

    # Convert lists to numpy arrays
    X = np.array(all_features)
    y = np.array(all_labels)

    print(f"\n--- Data Preparation Complete ---")
    print(f"Total Feature Vectors (Images): {X.shape[0]}")
    print(f"Feature Dimension: {X.shape[1] if X.shape[0] > 0 else 0}")

    return X, y


# --- STAGE 3 & 4: MODEL TRAINING AND EVALUATION ---

def train_behavior_classifier(X, y):
    """
    Trains a Random Forest classifier and prints the test accuracy.
    (This function is unchanged)
    """

    # 1. Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    print(f"\nTraining Model with {X_train.shape[0]} samples...")

    # 2. Initialize and Train the Model
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
    print("--- Starting Student Behavior AI Model Workflow (Image-Based) ---")

    # 1. LOAD AND PREPARE DATA (Feature Extraction)
    X_features, y_labels = load_and_prepare_data()

    if X_features.shape[0] == 0:
        print("\n[STOPPED] No data was loaded. Please ensure your images are in the correct directory structure:")
        # Updated print message to show all 4 folders
        print(f"  {DATA_DIR}/Attentive/image1.jpg")
        print(f"  {DATA_DIR}/Inattentive/image2.jpg")
        print(f"  {DATA_DIR}/Talking/image3.jpg")
        print(f"  {DATA_DIR}/Sleeping/image4.jpg")
    else:
        # 2. TRAIN CLASSIFIER MODEL
        trained_model = train_behavior_classifier(X_features, y_labels)

        # 3. Deployment Prep (Save the model for later use)
        # --- THIS IS NOW FIXED ---
        # The code will now save the model file, fixing your EOFError.
        joblib.dump(trained_model, 'student_behavior_model.pkl')
        print("\nModel saved as 'student_behavior_model.pkl'.")
        print("You can now run your 'run_live_detection.py' script.")