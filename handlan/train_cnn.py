import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# 데이터 불러오기
csv_path = 'hand_landmark_data.csv'
data = pd.read_csv(csv_path, header=None)

X = data.iloc[:, 1:].values.astype(np.float32)
y = data.iloc[:, 0].values

# 라벨 인코딩 및 one-hot
le = LabelEncoder()
y_encoded = le.fit_transform(y)
y_onehot = to_categorical(y_encoded)

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X, y_onehot, test_size=0.2, random_state=42)

# 모델 정의
model = Sequential([
    Dense(128, activation='relu', input_shape=(42,)),
    Dropout(0.2),
    Dense(64, activation='relu'),
    Dense(y_onehot.shape[1], activation='softmax')
])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 학습
model.fit(X_train, y_train, epochs=30, batch_size=16, validation_data=(X_test, y_test))

# 모델 저장
model.save('hand_sign_cnn.h5')
# 라벨 인코더 저장
import pickle
with open('label_encoder.pkl', 'wb') as f:
    pickle.dump(le, f) 