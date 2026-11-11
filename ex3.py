import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense

X_train = np.random.randn(270, 20, 12)
y_train = np.random.randint(0, 9, 270)
X_test = np.random.randn(54, 20, 12)
y_test = np.random.randint(0, 9, 54)

model = Sequential([
    LSTM(100, input_shape=(20, 12)),
    Dropout(0.5),
    Dense(9, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=5, batch_size=27, validation_data=(X_test, y_test), verbose=1)

loss, accuracy = model.evaluate(X_test, y_test)
print("Test Loss:", loss)
print("Test Accuracy:", accuracy)
