import cv2
import mediapipe as mp
import csv

video_path = 'youtube_videos/hello.mp4'
label = 'hello'
output_csv = 'hand_landmark_data.csv'

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1)

cap = cv2.VideoCapture(video_path)
csv_file = open(output_csv, 'w', newline='')
csv_writer = csv.writer(csv_file)

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
    if frame_count > 300:  # 300프레임(약 10초)까지만 저장
        break
csv_file.close()
cap.release()
print('손 랜드마크 추출 및 저장 완료!') 