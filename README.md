# HANDLAN: 실시간 수어(손동작) 인식 시스템

## 프로젝트 소개

**HANDLAN**은 MediaPipe와 LSTM 기반 딥러닝 모델을 활용하여 실시간으로 손동작(수어)을 인식하는 시스템입니다. 웹캠/영상에서 손 랜드마크를 추출하고, 시퀀스 데이터를 학습하여 hello, thankyou, goodbye 등 다양한 수어를 구분할 수 있습니다.

---

## 전체 파이프라인
1. **데이터 수집/추출**
   - 유튜브/mp4 영상에서 손 랜드마크 시퀀스 자동 추출
   - 직접 녹화(웹캠)로 시퀀스 데이터 수집
2. **데이터 전처리**
   - 시퀀스 길이(3초, 90프레임)로 통일
3. **LSTM 모델 학습**
   - 추출된 시퀀스 데이터로 LSTM 분류기 학습
4. **실시간 예측**
   - 웹캠에서 손동작을 3초간 녹화 후, 수어 예측 결과를 실시간 표시

---

## 주요 파일 설명

- `extract_sequence_from_videos.py` : mp4 영상에서 손 랜드마크 시퀀스(3초) 자동 추출
- `train_lstm.py` : 시퀀스 데이터로 LSTM 모델 학습 및 저장
- `realtime_lstm_inference.py` : 실시간 손동작 인식 및 예측(웹캠)
- `record_sequence.py`, `record_sequence_labeled.py` : 직접 녹화로 시퀀스 데이터 수집
- `youtube_crawl_and_label.py` : 유튜브에서 수어 영상 자동 다운로드
- `handlan/` : 파이프라인 전체 코드 및 유틸리티 모음

---

## 실행 환경
- Python 3.8 이상
- 필수 패키지: `opencv-python`, `mediapipe`, `tensorflow`, `scikit-learn`, `numpy`, `pandas`, `yt-dlp` 등
- 설치 예시:
  ```bash
  pip install opencv-python mediapipe tensorflow scikit-learn numpy pandas yt-dlp
  ```

---

## 사용법 예시

1. **영상에서 시퀀스 데이터 추출**
   ```bash
   python3 extract_sequence_from_videos.py
   ```
2. **LSTM 모델 학습**
   ```bash
   python3 handlan/train_lstm.py
   ```
3. **실시간 예측 실행**
   ```bash
   python3 realtime_lstm_inference.py
   ```

---

## 참고/기타
- 데이터셋, 모델, 코드 구조 등은 프로젝트 목적에 맞게 자유롭게 수정/확장 가능합니다.
- 문의: honggyeong (github.com/honggyeong) 