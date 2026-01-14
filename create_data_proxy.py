import os
import pickle
import mediapipe as mp
import cv2
import numpy as np
from sklearn.preprocessing import LabelEncoder

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=2,
    min_detection_confidence=0.5
)

DATA_DIR = 'data'
SEQUENCE_LENGTH = 30
FEATURE_SIZE = 126

data = []
labels = []

label_names = sorted(os.listdir(DATA_DIR))
label_encoder = LabelEncoder()

for label in label_names:
    img_dir = os.path.join(DATA_DIR, label)
    img_files = sorted(os.listdir(img_dir))

    # 🔥 Split 900 images → multiple sequences
    for i in range(0, len(img_files), SEQUENCE_LENGTH):
        sequence = []
        chunk = img_files[i:i + SEQUENCE_LENGTH]

        if len(chunk) < SEQUENCE_LENGTH:
            continue  # drop incomplete sequence

        for img_name in chunk:
            img_path = os.path.join(img_dir, img_name)
            frame = cv2.imread(img_path)
            if frame is None:
                continue

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(frame_rgb)

            frame_features = np.zeros(FEATURE_SIZE, dtype=np.float32)

            if results.multi_hand_landmarks and results.multi_handedness:
                hands_sorted = sorted(
                    zip(results.multi_handedness, results.multi_hand_landmarks),
                    key=lambda x: x[0].classification[0].label
                )

                idx = 0
                for _, hand_landmarks in hands_sorted:
                    for lm in hand_landmarks.landmark:
                        frame_features[idx:idx+3] = [lm.x, lm.y, lm.z]
                        idx += 3

            norm = np.linalg.norm(frame_features)
            if norm > 0:
                frame_features /= norm

            sequence.append(frame_features)

        data.append(sequence)
        labels.append(label)

labels = label_encoder.fit_transform(labels)

data = np.array(data, dtype=np.float32)
labels = np.array(labels, dtype=np.int64)

with open('data_seq1.pickle', 'wb') as f:
    pickle.dump({
        'data': data,
        'labels': labels,
        'label_map': label_encoder.classes_
    }, f)

print("Dataset created")
print("Data shape:", data.shape)
