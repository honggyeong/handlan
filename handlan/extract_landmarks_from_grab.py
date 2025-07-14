import os
import cv2
import mediapipe as mp
import pandas as pd
import csv

video_dir = 'grab_videos'
label_csv = 'grab_labels.csv'
output_csv = 'grab_landmark_data.csv'

labels_df = pd.read_csv(label_csv)
label_map = dict(zip(labels_df['filename'], labels_df['label']))

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1)

csv_file = open(output_csv, 'w', newline='')
csv_writer = csv.writer(csv_file)

for fname in os.listdir(video_dir):
    if not fname.endswith('.mp4'):
        continue
    label = label_map.get(fname)
    if not label:
        continue
    cap = cv2.VideoCapture(os.path.join(video_dir, fname))
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(img_rgb)
        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                landmarks = []
                for lm in hand_landmarks.landmark:
                    landmarks.append(lm.x)
                    landmarks.append(lm.y)
                if len(landmarks) == 42:
                    csv_writer.writerow([label] + landmarks)
        frame_count += 1
        if frame_count > 200:  # 영상당 200프레임까지만 저장
            break
    cap.release()

csv_file.close()
print('손 랜드마크 추출 및 저장 완료!') 