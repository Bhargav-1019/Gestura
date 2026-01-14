import cv2
import mediapipe as mp
import torch
import pickle
import numpy as np
from collections import deque, Counter
from train_proxy import GestureTransformer
import google.generativeai as genai
from deep_translator import GoogleTranslator
import pyttsx3
from gtts import gTTS
from dotenv import load_dotenv
import os

#GEMINI SETUP
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
gemini_model = genai.GenerativeModel('gemini-1.5-flash')

# SPEECH
def speak_text(text):
    try:
        engine = pyttsx3.init()
        engine.setProperty('rate', 150)
        engine.say(text)
        engine.runAndWait()
    except:
        tts = gTTS(text=text, lang='hi')
        tts.save("output.mp3")
        os.system("start output.mp3")

def process_and_speak(sentence):
    prompt = f"Correct the grammar of this sentence and return only the corrected sentence: '{sentence}'"
    response = gemini_model.generate_content(prompt)
    corrected = response.text.strip().strip('"')
    hindi = GoogleTranslator(source='auto', target='hi').translate(corrected)
    print("[Final]", hindi)
    speak_text(hindi)

# MODEL SETUP 
FEATURE_DIM = 126
SEQUENCE_LENGTH = 30

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = GestureTransformer(input_dim=126, seq_length=30)
model.load_state_dict(torch.load("gesture_transformer_proxy.pth", map_location=device))
model.eval()
model.to(device)

with open("data_seq1.pickle", "rb") as f:
    data_dict = pickle.load(f)

label_map = data_dict["label_map"]
print("LABEL MAP:", label_map)

# MEDIAPIPE
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5
)

# BUFFERS
sequence = deque(maxlen=SEQUENCE_LENGTH)
predictions = deque(maxlen=15)

sentence = []
collecting = False
last_added_word = ""

# CAMERA
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (640, 360))
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    # LANDMARK EXTRACTION 
    landmarks = np.zeros(FEATURE_DIM, dtype=np.float32)

    if results.multi_hand_landmarks and results.multi_handedness:
        hands_sorted = sorted(
            zip(results.multi_handedness, results.multi_hand_landmarks),
            key=lambda x: x[0].classification[0].label  # Left first
        )

        idx = 0
        for _, hand_landmarks in hands_sorted:
            for lm in hand_landmarks.landmark:
                landmarks[idx:idx+3] = [lm.x, lm.y, lm.z]
                idx += 3

    # normalize (SAME AS TRAINING)
    norm = np.linalg.norm(landmarks)
    if norm > 0:
        landmarks /= norm

    sequence.append(landmarks)

    label_to_display = "Detecting..."

    # PREDICTION
    if len(sequence) == SEQUENCE_LENGTH:
        x = torch.tensor(sequence, dtype=torch.float32).unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(x)
            probs = torch.softmax(logits, dim=1)
            conf, pred = torch.max(probs, dim=1)

        if conf.item() > 0.7:
            label = label_map[pred.item()]
            predictions.append(label)
        else:
            label = "Detecting..."

        # SMOOTHING
        if predictions:
            most_common, count = Counter(predictions).most_common(1)[0]
            label_to_display = most_common

            if collecting and count >= 4 and most_common != last_added_word:
                print("ADDING WORD:", most_common)
                sentence.append(most_common)
                last_added_word = most_common
                predictions = deque(maxlen=15)


    # DISPLAY
    cv2.putText(frame, label_to_display, (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    if collecting:
        cv2.putText(frame, "Sentence: " + " ".join(sentence), (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

    cv2.imshow("Sign Language Recognition", frame)

    # KEYS 
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break

    elif key == ord('s'):
        print("[INFO] Start sentence")
        collecting = True
        sentence = []
        last_added_word = ""
        predictions.clear()

    elif key == ord('e'):
        collecting = False
        final_sentence = " ".join(sentence)
        print("[Sentence]", final_sentence)
        if final_sentence:
            process_and_speak(final_sentence)
        sentence = []

cap.release()
cv2.destroyAllWindows()
