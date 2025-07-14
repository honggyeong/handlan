try:
    print('시작')
    import cv2
    import mediapipe as mp
    import numpy as np
    from tensorflow.keras.models import load_model
    import pickle
    import time
    print('import 완료')
    model = load_model('sign_lstm_model.h5')
    print('모델 로드 완료')
    with open('sign_lstm_label_encoder.pkl', 'rb') as f:
        le = pickle.load(f)
    print('라벨 인코더 로드 완료')
    seq_length = 90
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(max_num_hands=1)
    print('mediapipe 준비 완료')
    cap = cv2.VideoCapture(0)
    print('카메라 객체 생성')
    if not cap.isOpened():
        print('카메라를 열 수 없습니다. 카메라 연결/권한을 확인하세요.')
        exit()
    print('카메라 오픈 성공')
    print('실시간 예측 루프 진입')
    recording = False
    current_seq = []
    prediction = ''
    pred_time = 0
    prev_space = False
    print('스페이스바를 누르는 동안 동작을 녹화, 뗐을 때 예측 결과를 화면에 표시합니다. ESC로 종료')
    while True:
        ret, frame = cap.read()
        if not ret:
            print('프레임 읽기 실패')
            break
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(img_rgb)
        landmarks = []
        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                for lm in hand_landmarks.landmark:
                    landmarks.append(lm.x)
                    landmarks.append(lm.y)
        key = cv2.waitKey(1) & 0xFF
        now = time.time()
        space_pressed = (key == 32)
        if space_pressed and not prev_space:
            recording = True
            current_seq = []
            print('녹화 시작')
        elif not space_pressed and prev_space:
            if len(current_seq) >= 1:
                seq = current_seq.copy()
                if len(seq) < seq_length:
                    pad = [seq[-1]] * (seq_length - len(seq))
                    seq = seq + pad
                else:
                    seq = seq[-seq_length:]
                seq = np.array(seq).reshape(1, seq_length, 42)
                pred = model.predict(seq)
                class_id = np.argmax(pred)
                prediction = le.inverse_transform([class_id])[0]
                pred_time = now
                print(f'예측 결과: {prediction}')
            else:
                prediction = 'Too short'
                pred_time = now
                print('녹화된 시퀀스가 너무 짧음')
            recording = False
            current_seq = []
        prev_space = space_pressed
        if recording and len(landmarks) == 42:
            current_seq.append(landmarks)
        if prediction and now - pred_time < 2:
            color = (0,255,0) if prediction != 'Too short' else (0,0,255)
            cv2.putText(frame, f'Prediction: {prediction}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        if recording:
            cv2.putText(frame, f'Recording... ({len(current_seq)})', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
        cv2.imshow('Realtime LSTM Inference', frame)
        if key == 27:
            print('ESC 입력, 종료')
            break
    cap.release()
    cv2.destroyAllWindows()
    print('종료')
except Exception as e:
    print('예외 발생:', e) 