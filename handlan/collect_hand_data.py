import cv2
import mediapipe as mp
import numpy as np
import csv

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

label = input('저장할 수어 라벨을 입력하세요: ')

csv_file = open('hand_landmark_data.csv', 'a', newline='')
csv_writer = csv.writer(csv_file)

print("s 키를 누르면 데이터가 저장됩니다. esc로 종료")

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
    cv2.imshow('Collect Hand Data', frame)
    key = cv2.waitKey(1)
    if key & 0xFF == ord('s'):
        if len(landmarks) == 42:
            csv_writer.writerow([label] + landmarks)
            print('저장됨!')
        else:
            print('손이 인식되지 않았습니다.')
    if key & 0xFF == 27:
        break

csv_file.close()
cap.release()
cv2.destroyAllWindows() 