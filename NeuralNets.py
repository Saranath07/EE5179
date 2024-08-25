import numpy as np
from activations import Activations
from lossfunctions import LossFunctions
from sklearn.metrics import r2_score
from initialization import ParameterInitializer

softmax = Activations.Softmax()

sigmoid = Activations.Sigmoid()

crossentropy = LossFunctions.CrossEntropy()



class FeedForwardNeuralNets:
    def __init__(self, inputs, hidden_layers, outputs, g=sigmoid, L=crossentropy, O=softmax, eta=0.01, optimizer="gd", initialization_method="gaussian"):
        self.inputs = inputs
        self.outputs = outputs
        self.parameters = ParameterInitializer(initialization_method).initialize_parameters(inputs[0].shape[0], hidden_layers, outputs[0].shape[0])
        self.g = g
        self.O = O
        self.L = L
        self.eta = eta
        self.losses = {}
        self.activations = {}
        if optimizer == "gd":
            self.optimizer = self.gradient_descent
        elif optimizer == "sgd":
            self.optimizer = self.sgd
        

    def forward_propogation(self, x):
        self.activations["a1"] = self.parameters["W1"] @ x + self.parameters["b1"]
        self.activations["h1"] = self.g(self.activations["a1"])
        for i in range(2, len(self.parameters) // 2):
            self.activations[f"a{i}"] = self.parameters[f"W{i}"] @ self.activations[f"h{i - 1}"] + self.parameters[f"b{i}"]
            self.activations[f"h{i}"] = self.g(self.activations[f"a{i}"])

        self.activations[f"a{len(self.parameters) // 2}"] = self.parameters[f"W{len(self.parameters) // 2}"] @ self.activations[f"h{len(self.parameters) // 2 - 1}"] + self.parameters[f"b{len(self.parameters) // 2}"]
        y_pred = self.O(self.activations[f"a{len(self.parameters) // 2}"])
        # print(y_pred)
        return y_pred

    def backPropogation(self, y_pred, y, x):
        n = len(self.parameters) // 2
        m = len(self.activations) // 2
        La = y_pred - y

        Lh = La @ self.parameters[f"W{n}"]

        da = self.g.gradient(self.activations[f"a{m}"])
        self.losses[f"W{n}"] = np.outer(La, self.activations[f"h{m}"])
        self.losses[f"b{n}"] = La.copy()

        for i in range(1, m):
            La = Lh * da
            Lh = La @ self.parameters[f"W{m - i + 1}"]
            da = self.g.gradient(self.activations[f"a{m - i}"])
            self.losses[f"W{m - i + 1}"] = np.outer(La, self.activations[f"h{m - i}"])
            self.losses[f"b{m - i + 1}"] = La.copy()

        La = Lh * da
        self.losses["W1"] = np.outer(La, x)
        self.losses["b1"] = La.copy()

    def gradient_descent(self):
        for i in range(len(self.parameters) // 2):
            self.parameters[f"W{i + 1}"] = self.parameters[f"W{i + 1}"] - self.eta * self.losses[f"W{i + 1}"]
            self.parameters[f"b{i + 1}"] = self.parameters[f"b{i + 1}"] - self.eta * self.losses[f"b{i + 1}"]
        

    def sgd(self):
        
        num_samples = len(self.losses)  # Number of data points or samples

        # Iterate over each sample and update parameters using its corresponding gradient
        for i in range(num_samples):
            for j in range(1, len(self.parameters) // 2 + 1):  # Iterate over layers
                self.parameters[f"W{j}"] = self.parameters[f"W{j}"] - self.eta * self.losses[f"W{j}"]
                self.parameters[f"b{j}"] = self.parameters[f"b{j}"] - self.eta * self.losses[f"b{j}"]

        

    def train(self, epochs):
        for _ in range(epochs):
            # print(self.parameters)
            total_gradient_loss = {key: 0 for key in self.parameters.keys()}  # Initialize total gradient loss
            for x, y in zip(self.inputs, self.outputs):  # Iterate over each data point
                y_pred = self.forward_propogation(x)
                self.backPropogation(y_pred, y, x)
                # Accumulate the gradients for each data point
                for key in total_gradient_loss:
                    total_gradient_loss[key] += self.losses.get(key, 0)
            # Update the parameters using the accumulated gradients
            self.optimizer()

    def evaluate(self, X):
        return np.array(self.forward_propogation(X))
