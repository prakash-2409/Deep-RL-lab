import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

class DeepNeuralNetwork:
    def __init__(self, input_size, hidden_layers, output_size, task):
        self.input_size = input_size
        self.hidden_layers = hidden_layers
        self.output_size = output_size
        self.task = task
        self.model = self.build_model()

    def build_model(self):
        model = Sequential()
        model.add(Dense(self.hidden_layers[0], input_dim=self.input_size, activation='relu'))
        for layer_size in self.hidden_layers[1:]:
            model.add(Dense(layer_size, activation='relu'))

        if self.task == 'classification':
            model.add(Dense(self.output_size, activation='softmax'))
            model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        elif self.task == 'regression':
            model.add(Dense(self.output_size, activation='linear'))
            model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mse'])
        else:
            raise ValueError("Invalid task. Choose 'classification' or 'regression'.")
        return model

    def train(self, X_train, y_train, epochs, batch_size):
        if self.task == 'classification':
            y_train_categorical = np.eye(self.output_size)[y_train]
            self.model.fit(X_train, y_train_categorical, epochs=epochs, batch_size=batch_size, verbose=0)
        elif self.task == 'regression':
            self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1)

    def predict(self, X_test):
        return self.model.predict(X_test)

X_classification = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_classification = np.array([0, 1, 1, 0])

X_regression = np.array([[1], [2], [3], [4], [5]])
y_regression = np.array([2, 4, 6, 8, 10])

hidden_layers = [4, 3]
epochs = 100
batch_size = 1

dnn_classification = DeepNeuralNetwork(X_classification.shape[1], hidden_layers, len(np.unique(y_classification)), 'classification')
dnn_classification.train(X_classification, y_classification, epochs, batch_size)

dnn_regression = DeepNeuralNetwork(X_regression.shape[1], hidden_layers, 1, 'regression')
dnn_regression.train(X_regression, y_regression, epochs, batch_size)

print("Predicted output for classification task:")
print(dnn_classification.predict(np.array([[0, 0], [0, 1], [1, 0], [1, 1]])))

print("Predicted output for regression task:")
print(dnn_regression.predict(np.array([[6], [7], [8]])))
