import cv2
import mediapipe as mp
import numpy as np

label = input('수어 라벨을 입력하세요: ')
save_path = f'{label}_sequences.npy'
sequences = []

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
cap = cv2.VideoCapture(0)

print('스페이스바를 누르고 있으면 녹화, 뗐다가 다시 누르면 새로운 시퀀스 저장, ESC로 종료')

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
    cv2.imshow('Record Sequence', frame)
    key = cv2.waitKey(1)
    if key == 27:  # ESC
        break
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
            sequences.append(current_seq)
            recording = False

cap.release()
cv2.destroyAllWindows()
np.save(save_path, np.array(sequences, dtype=np.float32))
print(f'시퀀스 {len(sequences)}개 저장 완료: {save_path}') 