import os
import cv2
import numpy as np
import mediapipe as mp

# Configuration
DATA_DIR = "tennis_dataset_videos"  # Folder containing subfolders per action (e.g., /serve, /forehand)
SEQUENCE_LENGTH = 30  # Number of frames tracked per stroke sequence
X_DATA = []
Y_DATA = []

# Map classes to integers
LABEL_MAP = {"serve": 0, "forehand": 1, "backhand": 2, "volley": 3}

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.6, min_tracking_confidence=0.6)

print("Starting video pre-processing and feature extraction...")

for action_name, action_label in LABEL_MAP.items():
    action_folder = os.path.join(DATA_DIR, action_name)
    if not os.path.exists(action_folder):
        continue
        
    for video_file in os.listdir(action_folder):
        video_path = os.path.join(action_folder, video_file)
        cap = cv2.VideoCapture(video_path)
        
        window_sequence = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            # Convert color space for MediaPipe processing
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(image_rgb)
            
            if results.pose_landmarks:
                # Flatten the x, y, z, and visibility coordinates of all 33 tracking nodes
                frame_landmarks = []
                for lm in results.pose_landmarks.landmark:
                    frame_landmarks.extend([lm.x, lm.y, lm.z, lm.visibility])
                
                window_sequence.append(frame_landmarks)
                
                # Once we gather a complete sliding sequence window, lock it as a training point
                if len(window_sequence) == SEQUENCE_LENGTH:
                    X_DATA.append(window_sequence)
                    Y_DATA.append(action_label)
                    window_sequence = window_sequence[1:]  # Slide window forward by 1 frame
                    
        cap.release()

# Convert structured sequences to high performance numpy structures
X = np.array(X_DATA)
y = np.array(Y_DATA)

# Save arrays to file arrays for instant training reloading
np.save("X_sequences.npy", X)
np.save("y_labels.npy", y)
print(f"Data ready. Extracted {X.shape[0]} unique structural frame sequences.")