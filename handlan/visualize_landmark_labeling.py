import cv2
import mediapipe as mp

# 영상 파일과 라벨 지정 (필요시 아래 두 줄만 수정)
video_path = 'youtube_videos/goodbye.mp4'  # 또는 'youtube_videos/hello.mp4', 'thankyou.mp4'
label = 'goodbye'  # 또는 'hello', 'thankyou'

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(video_path)
frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(img_rgb)
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        cv2.putText(frame, f'Label: {label}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    else:
        cv2.putText(frame, 'No hand detected', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
    cv2.imshow('Sign Video - Landmark Visualization', frame)
    key = cv2.waitKey(30)  # 30ms 간격(약 33fps)
    if key == 27:  # ESC로 종료
        break
    frame_count += 1
    if frame_count > 300:  # 300프레임(약 10초)까지만
        break

cap.release()
cv2.destroyAllWindows() 