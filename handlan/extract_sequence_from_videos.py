import os
import cv2
import mediapipe as mp
import numpy as np

video_label_map = {
    'hello_SsLvqfTXo78.mp4': 'hello',
    'thankyou.mp4': 'thankyou',
    'goodbye.mp4': 'goodbye'
}
video_dir = 'youtube_videos'
seq_length = 30  # 시퀀스 길이(프레임 수)
all_sequences = []
all_labels = []

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1)

for fname, label in video_label_map.items():
    path = os.path.join(video_dir, fname)
    if not os.path.exists(path):
        print(f'{path} 파일이 없습니다.')
        continue
    cap = cv2.VideoCapture(path)
    frames = []
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
        if len(landmarks) == 42:
            frames.append(landmarks)
    cap.release()
    # 시퀀스 분할
    for i in range(0, len(frames) - seq_length + 1, seq_length):
        seq = frames[i:i+seq_length]
        if len(seq) == seq_length:
            all_sequences.append(seq)
            all_labels.append(label)

np.save('sign_sequences.npy', np.array(all_sequences, dtype=np.float32))
np.save('sign_labels.npy', np.array(all_labels))
print(f'총 {len(all_sequences)}개 시퀀스 저장 완료!') 