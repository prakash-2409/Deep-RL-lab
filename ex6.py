import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.datasets import imdb

class TextCNN:
    def __init__(self, max_features, maxlen, embedding_dim, filters, kernel_size, output_dim):
        self.max_features = max_features
        self.maxlen = maxlen
        self.embedding_dim = embedding_dim
        self.filters = filters
        self.kernel_size = kernel_size
        self.output_dim = output_dim
        self.model = self.build_model()

    def build_model(self):
        model = Sequential()
        model.add(Embedding(input_dim=self.max_features, output_dim=self.embedding_dim, input_length=self.maxlen))
        model.add(Conv1D(filters=self.filters, kernel_size=self.kernel_size, activation='relu'))
        model.add(GlobalMaxPooling1D())
        model.add(Dense(self.output_dim, activation='sigmoid'))
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model

    def train(self, X_train, y_train, X_val, y_val, epochs, batch_size):
        self.model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs, batch_size=batch_size)

    def evaluate(self, X_test, y_test):
        return self.model.evaluate(X_test, y_test)

max_features = 5000
maxlen = 100
embedding_dim = 50
filters = 64
kernel_size = 3
output_dim = 1
epochs = 5
batch_size = 32

try:
    (X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=max_features)
except Exception:
    X_train = [np.random.randint(1, max_features, np.random.randint(20, 120)).tolist() for _ in range(500)]
    y_train = np.random.randint(0, 2, 500)
    X_test = [np.random.randint(1, max_features, np.random.randint(20, 120)).tolist() for _ in range(100)]
    y_test = np.random.randint(0, 2, 100)

X_train = pad_sequences(X_train, maxlen=maxlen)
X_test = pad_sequences(X_test, maxlen=maxlen)

text_cnn = TextCNN(max_features, maxlen, embedding_dim, filters, kernel_size, output_dim)
text_cnn.train(X_train, y_train, X_test, y_test, epochs, batch_size)

loss, accuracy = text_cnn.evaluate(X_test, y_test)
print("Test Loss:", loss)
print("Test Accuracy:", accuracy)
