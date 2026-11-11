import numpy as np
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense

googlenet = InceptionV3(weights=None, classes=1000)
print("GoogLeNet (InceptionV3) created. Total layers:", len(googlenet.layers))

rnn = Sequential([
    SimpleRNN(10, input_shape=(10, 1), return_sequences=True),
    Dense(1)
])
rnn.compile(optimizer='adam', loss='mean_squared_error')
X = np.random.randn(50, 10, 1)
y = np.random.randn(50, 10, 1)
rnn.fit(X, y, epochs=5, verbose=1)
print("RNN Model trained successfully.")
