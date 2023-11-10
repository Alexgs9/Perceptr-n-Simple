# Perceptron simple usando funcion de activacion sigmoidal
# Autor: Alexandro Gutiérrez Serna

import numpy as np

class Perceptron:
    def __init__(self, input_size):
        self.weights = np.random.rand(input_size)
        self.bias = 1
        self.learning_rate = 0.01

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def predict(self, inputs):
        activation = np.dot(self.weights, inputs) + self.bias
        prediction = self.sigmoid(activation)
        return 1 if prediction >= 0.5 else 0

    def train(self, inputs, target, epochs):
        for epoch in range(epochs):
            #print(f"\nEpoch {epoch + 1}/{epochs}")
            for i in range(len(inputs)):
                prediction = self.predict(inputs[i])
                error = target[i] - prediction

                # Imprimir el proceso de ajuste de pesos
                #print(f"  Input: {inputs[i]}, Target: {target[i]}, Prediction: {prediction}, Error: {error}")
                #print(f"  Weights before: {self.weights}, Bias before: {self.bias}")

                self.weights += self.learning_rate * error * inputs[i]
                self.bias += self.learning_rate * error

        # Imprimir los pesos y el sesgo después del ajuste
        print(f"  Weights after: {self.weights}, Bias after: {self.bias}")

    def accuracy(self, inputs, targets):
        correct_predictions = 0
        total_samples = len(inputs)

        for i in range(total_samples):
            prediction = self.predict(inputs[i])
            if prediction == targets[i]:
                correct_predictions += 1

        accuracy = correct_predictions / total_samples
        return accuracy


input_size = 2
epochs = 100
perceptron = Perceptron(input_size)

# ----------------------- Datos de entrenamiento (AND gate) ----------------------- 
inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
targets = np.array([0, 0, 0, 1])

perceptron.train(inputs, targets, epochs)

# Prueba del perceptrón entrenado
for i in range(len(inputs)):
    prediction = perceptron.predict(inputs[i])
    print(f'Input: {inputs[i]}, Prediction: {prediction}')

# Calcular la exactitud en los datos de prueba
accuracy = perceptron.accuracy(inputs, targets)

print(f'Exactitud en los datos de prueba: {accuracy * 100:.2f}%')
print("\n")
# ----------------------- Datos de entrenamiento (OR gate) ----------------------- 
perceptron_or = Perceptron(input_size)

inputs_or = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
targets_or = np.array([0, 1, 1, 1])

# Entrenamiento con epochs aleatorios
perceptron_or.train(inputs_or, targets_or, epochs)

# Prueba del perceptrón entrenado con datos de entrenamiento
print("Prueba con datos de entrenamiento para OR gate:")
for i in range(len(inputs_or)):
    prediction = perceptron_or.predict(inputs_or[i])
    print(f'Input: {inputs_or[i]}, Prediction: {prediction}')

# Calcular la exactitud en los datos de prueba
accuracy = perceptron.accuracy(inputs, targets)

print(f'Exactitud en los datos de prueba: {accuracy * 100:.2f}%')