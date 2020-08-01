import numpy as np

class NeuralNetwork():

    def __init__(self):
        np.random.seed(1)

        # initialise weights
        self.synaptic_weights = 2 * np.random.random((3, 1)) - 1

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))  # 1/(1+e^-x)

    def sigmoid_derivative (self, x):
        return x * (1 - x)

    def train(self, training_inputs, actual_outputs, training_iterations):

        for iteration in range(training_iterations):

            output = self.think(training_inputs)
            error = actual_outputs - output

            adjustments = error * self.sigmoid_derivative(output)

            self.synaptic_weights += (training_inputs.T).dot(adjustments)

    def think(self, inputs):

        inputs = inputs.astype(float)
        output = self.sigmoid(inputs.dot(self.synaptic_weights))

        return output


if __name__ == "__main__":

    neural_network = NeuralNetwork()

    print("Random Synaptic Weights: ")
    print(neural_network.synaptic_weights)

    training_inputs = np.array([[0, 0, 1],
                                [1, 1, 1],
                                [1, 0, 1],
                                [0, 1, 1]
                                ])

    # if the first input is 0 output is 0 and if 1st input is 1 output should be 1
    actual_outputs = np.array([[0, 1, 1, 0]]).T

    neural_network.train(training_inputs, actual_outputs, 10000)

    print("Synaptic Weights After Training")
    print(neural_network.synaptic_weights)

    A = input("Input 1: ")
    B = input("Input 2: ")
    C = input("Input 3: ")

    print("Test Data: input data = ", A, B, C)
    print("Output Data: ")
    print(neural_network.think(np.array([A, B, C])))

