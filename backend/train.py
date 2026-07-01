import os
import glob
import cv2
import numpy as np
import pandas as pd
import mediapipe as mp
import joblib  # Used to save our model
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Initialize MediaPipe Pose Core for Static Images
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)

def build_dataset_from_kaggle_images(base_dir):
    target_classes = ['forehand', 'backhand', 'serve']
    extracted_dataset = []

    for label_idx, folder_name in enumerate(target_classes):
        folder_path = os.path.join(base_dir, folder_name)
        if not os.path.exists(folder_path):
            print(f"⚠️ Folder not found, skipping: {folder_name}")
            continue

        print(f"📸 Scanning Kaggle images in folder: [{folder_name}]...")
        
        # Look for .jpg, .jpeg, and .png files
        image_files = (
            glob.glob(os.path.join(folder_path, "*.jpg")) + 
            glob.glob(os.path.join(folder_path, "*.jpeg")) + 
            glob.glob(os.path.join(folder_path, "*.png"))
        )
        
        for img_path in image_files:
            frame = cv2.imread(img_path)
            if frame is None:
                continue
                
            rgb_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = pose.process(rgb_img)

            if res.pose_landmarks:
                lm = res.pose_landmarks.landmark
                # Extract: Right Shoulder(12), Right Elbow(14), Right Wrist(16)
                extracted_dataset.append([
                    lm[12].x, lm[12].y,
                    lm[14].x, lm[14].y,
                    lm[16].x, lm[16].y,
                    label_idx  # Class ID
                ])

    columns = ['rs_x', 'rs_y', 're_x', 're_y', 'rw_x', 'rw_y', 'label_id']
    return pd.DataFrame(extracted_dataset, columns=columns)

if __name__ == "__main__":
    print("🚀 Initializing Image Feature Processing Engine...")
    dataset_path = "./KAGGLE_DATASET" 
    df = build_dataset_from_kaggle_images(dataset_path)

    if df.empty:
        print("❌ Data extract failed! Check your folder layout paths.")
        exit()

    X = df.drop('label_id', axis=1)
    y = df['label_id']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("🏋️‍♂️ Training final Random Forest Classifier...")
    # Swapped from XGBoost to Random Forest to fix the Mac libomp error!
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    os.makedirs("models", exist_ok=True)
    
    # Save the model weights file where app.py can read it
    joblib.dump(model, "models/tennis_sequence_model.pkl")
    print(f"\n✅ Done! Accuracy Score: {model.score(X_test, y_test)*100:.2f}%")