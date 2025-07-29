import cv2
import mediapipe as mp
import numpy as np
import pickle

with open("sign_model.pkl", "rb") as f:
    model = pickle.load(f)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2)
prev_letter = ""
stable_count = 0
stable_threshold = 15

def predict_from_frame(image, current_word):
    global prev_letter, stable_count

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    current_letter = ""
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            data = []
            for lm in hand_landmarks.landmark:
                data.extend([lm.x, lm.y, lm.z])
            data = np.array(data).reshape(1, -1)

            prediction = model.predict(data)[0]
            current_letter = prediction

            if prediction == prev_letter:
                stable_count += 1
            else:
                stable_count = 0
            prev_letter = prediction

            if stable_count >= stable_threshold:
                current_word += prediction
                stable_count = 0

    return current_letter, current_word