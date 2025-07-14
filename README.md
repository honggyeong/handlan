# HANDLAN: 실시간 수어(손동작) 인식 시스템

## 프로젝트 소개

**HANDLAN**은 청각장애인과 비장애인 간의 소통을 돕기 위해, 컴퓨터 비전과 인공지능 기술을 활용하여 손동작(수어, 즉 손으로 하는 언어)을 실시간으로 인식하는 시스템입니다.

이 프로젝트는 동아리 주제 탐구 활동의 일환으로 진행되었으며, 학교 선생님과 학생 누구나 쉽게 이해하고 활용할 수 있도록 구성하였습니다.

---

## 프로젝트 목적
- **수어(손동작) 인식**을 통해 청각장애인의 의사소통을 지원
- 인공지능, 딥러닝, 컴퓨터 비전 등 최신 IT 기술을 직접 체험
- 영상 데이터 수집, 모델 학습, 실시간 예측 등 전체 인공지능 파이프라인 경험

---

## 기대 효과
- 수어를 모르는 사람도 손동작만으로 컴퓨터와 소통 가능
- 인공지능 모델의 원리와 실제 적용 과정을 쉽게 체험
- 동아리/학교 내 인공지능 교육 및 체험 활동 자료로 활용

---

## 동작 흐름(한눈에 보기)

1. **손동작 데이터 수집**
   - 유튜브/직접 촬영한 영상을 이용해 손의 움직임(랜드마크) 데이터를 수집합니다.
2. **데이터 전처리 및 학습**
   - 손동작 데이터를 인공지능 모델(LSTM)에 학습시킵니다.
3. **실시간 예측**
   - 웹캠으로 손동작을 3초간 녹화하면, 컴퓨터가 어떤 수어(hello, thankyou, goodbye 등)인지 실시간으로 예측해줍니다.

---

## 주요 기술 및 도구
- **MediaPipe**: 손의 21개 주요 관절 위치(랜드마크) 추출
- **TensorFlow/Keras**: LSTM(순환 신경망) 기반 딥러닝 모델 학습
- **OpenCV**: 웹캠 영상 처리 및 실시간 화면 표시
- **Python**: 전체 파이프라인 구현

---

## 주요 파일 설명

- `extract_sequence_from_videos.py` : 영상에서 손동작(3초) 시퀀스 자동 추출
- `train_lstm.py` : 손동작 시퀀스 데이터로 LSTM 모델 학습
- `realtime_lstm_inference.py` : 웹캠으로 실시간 손동작 인식 및 예측
- `record_sequence.py`, `record_sequence_labeled.py` : 직접 손동작 녹화 및 데이터 저장
- `youtube_crawl_and_label.py` : 유튜브에서 수어 영상 자동 다운로드
- `handlan/` : 파이프라인 전체 코드 및 유틸리티 모음

---

## 실행 환경 및 준비 방법
- Python 3.8 이상
- 필수 패키지: `opencv-python`, `mediapipe`, `tensorflow`, `scikit-learn`, `numpy`, `pandas`, `yt-dlp` 등
- 설치 예시:
  ```bash
  pip install opencv-python mediapipe tensorflow scikit-learn numpy pandas yt-dlp
  ```

---

## 사용법(예시)

1. **영상에서 손동작 데이터 추출**
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

## 활용 예시
- 동아리 시간에 직접 손동작을 녹화해보고, 인공지능이 어떤 수어인지 맞추는 체험
- 다양한 수어(hello, thankyou, goodbye 등)를 추가해보고, 모델의 성능 변화를 실험
- 영상 데이터(유튜브 등)로 데이터셋을 확장하여 더 많은 수어 인식 도전

---

## 참고자료 및 학습 링크
- [MediaPipe 공식 문서](https://google.github.io/mediapipe/solutions/hands.html)
- [TensorFlow 공식 문서](https://www.tensorflow.org/)
- [OpenCV 공식 문서](https://opencv.org/)
- [유튜브 수어 데이터셋 예시: WLASL](https://github.com/dxli94/WLASL)

---

## 문의 및 기여
- 프로젝트 문의: honggyeong (github.com/honggyeong)
- 누구나 자유롭게 코드/데이터를 수정·확장하여 활용할 수 있습니다.

---

> **이 프로젝트는 동아리 주제 탐구 및 인공지능 체험 교육을 위해 제작되었습니다.**
> 
> 학교 선생님, 학생 모두가 쉽게 따라할 수 있도록 최대한 친절하게 설명하였으니, 궁금한 점은 언제든 문의해 주세요! 