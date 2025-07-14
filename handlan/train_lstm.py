import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pickle

# 데이터 불러오기
X = np.load('sign_sequences.npy')  # (샘플수, 시퀀스길이, 42)
y = np.load('sign_labels.npy')     # (샘플수,)

# 라벨 인코딩 및 one-hot
le = LabelEncoder()
y_encoded = le.fit_transform(y)
y_onehot = to_categorical(y_encoded)

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X, y_onehot, test_size=0.2, random_state=42)

# LSTM 모델 정의
model = Sequential([
    LSTM(64, return_sequences=False, input_shape=(X.shape[1], X.shape[2])),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dense(y_onehot.shape[1], activation='softmax')
])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 학습
model.fit(X_train, y_train, epochs=30, batch_size=8, validation_data=(X_test, y_test))

# 모델 저장
model.save('sign_lstm_model.h5')
with open('sign_lstm_label_encoder.pkl', 'wb') as f:
    pickle.dump(le, f)
print('LSTM 모델 및 라벨 인코더 저장 완료!') 