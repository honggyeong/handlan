import cv2
import mediapipe as mp
import numpy as np
import os

save_path = 'custom_sequences.npz'
all_sequences = []
all_labels = []

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
cap = cv2.VideoCapture(0)

print('수어 라벨을 입력하고, 스페이스바를 누르고 있으면 녹화, 뗐다가 다시 누르면 새로운 시퀀스 저장, ESC로 종료')

while True:
    label = input('수어 라벨을 입력하세요 (종료하려면 엔터): ').strip()
    if label == '':
        break
    print(f'[{label}] 녹화 준비. 스페이스바를 누르고 있으면 녹화, 뗐다가 다시 누르면 시퀀스 저장')
    recording = False
    current_seq = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(img_rgb)
        landmarks = []
        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                for lm in hand_landmarks.landmark:
                    landmarks.append(lm.x)
                    landmarks.append(lm.y)
        cv2.putText(frame, f'Label: {label}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        cv2.imshow('Record Sequence', frame)
        key = cv2.waitKey(1)
        if key == 27:  # ESC
            cap.release()
            cv2.destroyAllWindows()
            np.savez(save_path, sequences=np.array(all_sequences, dtype=np.float32), labels=np.array(all_labels))
            print(f'총 {len(all_sequences)}개 시퀀스 저장 완료: {save_path}')
            exit(0)
        if key == 32:  # 스페이스바
            if not recording:
                print('녹화 시작')
                recording = True
                current_seq = []
            if len(landmarks) == 42:
                current_seq.append(landmarks)
        else:
            if recording and current_seq:
                print('녹화 종료, 시퀀스 저장')
                all_sequences.append(current_seq)
                all_labels.append(label)
                recording = False
                break  # 한 시퀀스 녹화 후 다음 라벨 입력으로 이동

cap.release()
cv2.destroyAllWindows()
np.savez(save_path, sequences=np.array(all_sequences, dtype=np.float32), labels=np.array(all_labels))
print(f'총 {len(all_sequences)}개 시퀀스 저장 완료: {save_path}') 