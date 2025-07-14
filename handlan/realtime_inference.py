import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model
import pickle

# 모델 및 라벨 인코더 불러오기
model = load_model('hand_sign_cnn.h5')
with open('label_encoder.pkl', 'rb') as f:
    le = pickle.load(f)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(img_rgb)
    landmarks = []
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            for lm in hand_landmarks.landmark:
                landmarks.append(lm.x)
                landmarks.append(lm.y)
    # 예측
    if len(landmarks) == 42:
        input_data = np.array(landmarks).reshape(1, -1)
        pred = model.predict(input_data)
        class_id = np.argmax(pred)
        class_name = le.inverse_transform([class_id])[0]
        cv2.putText(frame, f'Prediction: {class_name}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    cv2.imshow('Realtime Hand Sign Recognition', frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows() 