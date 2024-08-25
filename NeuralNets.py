import numpy as np
from activations import Activations
from lossfunctions import LossFunctions
from initialization import ParameterInitializer

softmax = Activations.Softmax()
sigmoid = Activations.Sigmoid()
crossentropy = LossFunctions.CrossEntropyLoss()

class FeedForwardNeuralNets:
    def __init__(self, inputs, hidden_layers, outputs, g=sigmoid, L=crossentropy, O=softmax, eta=0.01, 
                 optimizer="gd", initialization_method="gaussian", batch_size=32,
                 beta1=0.9, beta2=0.999, epsilon=1e-8, t=0):
        self.inputs = inputs
        self.outputs = outputs
        self.batch_size = min(batch_size, inputs.shape[0])
        if len(self.outputs.shape) < 2:
            self.parameters = ParameterInitializer(initialization_method).initialize_parameters(
            inputs[0].shape[0], hidden_layers, 1)
        else:
            self.parameters = ParameterInitializer(initialization_method).initialize_parameters(
                inputs[0].shape[0], hidden_layers, outputs[0].shape[0])
        self.g = g
        self.O = O
        self.L = L
        self.eta = eta
        self.losses = {}
        self.activations = {}
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.t = t
        self.v = {key: np.zeros_like(value) for key, value in self.parameters.items()}
        self.s = {key: np.zeros_like(value) for key, value in self.parameters.items()}
        
        if optimizer == "gd":
            self.optimizer = self.gradient_descent
        elif optimizer == "sgd":
            self.optimizer = self.sgd
        elif optimizer == "adam":
            self.optimizer = self.adam

    def forward_propogation(self, x):
        
        self.activations["a1"] = self.parameters["W1"] @ x + self.parameters["b1"]
        self.activations["h1"] = self.g(self.activations["a1"])
        for i in range(2, len(self.parameters) // 2):
            self.activations[f"a{i}"] = self.parameters[f"W{i}"] @ self.activations[f"h{i - 1}"] + self.parameters[f"b{i}"]
            self.activations[f"h{i}"] = self.g(self.activations[f"a{i}"])

        self.activations[f"a{len(self.parameters) // 2}"] = self.parameters[f"W{len(self.parameters) // 2}"] @ self.activations[f"h{len(self.parameters) // 2 - 1}"] + self.parameters[f"b{len(self.parameters) // 2}"]
        y_pred = self.O(self.activations[f"a{len(self.parameters) // 2}"])
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
        num_samples = len(self.losses)
        for i in range(num_samples):
            for j in range(1, len(self.parameters) // 2 + 1):
                self.parameters[f"W{j}"] = self.parameters[f"W{j}"] - self.eta * self.losses[f"W{j}"]
                self.parameters[f"b{j}"] = self.parameters[f"b{j}"] - self.eta * self.losses[f"b{j}"]

    def adam(self):
        self.t += 1
        for key in self.parameters.keys():
            if key.startswith("W") or key.startswith("b"):
                gradient = self.losses[key]

                self.v[key] = self.beta1 * self.v[key] + (1 - self.beta1) * gradient
                self.s[key] = self.beta2 * self.s[key] + (1 - self.beta2) * (gradient ** 2)

                v_corrected = self.v[key] / (1 - self.beta1 ** self.t)
                s_corrected = self.s[key] / (1 - self.beta2 ** self.t)

                self.parameters[key] -= self.eta * v_corrected / (np.sqrt(s_corrected) + self.epsilon)

    def train(self, epochs):
        for _ in range(epochs):
            permutation = np.random.permutation(self.inputs.shape[0])
            inputs_shuffled = self.inputs[permutation]
            outputs_shuffled = self.outputs[permutation]
            
            for i in range(0, self.inputs.shape[0], self.batch_size):
                batch_inputs = inputs_shuffled[i:i + self.batch_size]
                batch_outputs = outputs_shuffled[i:i + self.batch_size]

                total_gradient_loss = {key: 0 for key in self.parameters.keys()}
                for x, y in zip(batch_inputs, batch_outputs):
                    y_pred = self.forward_propogation(x)
                    self.backPropogation(y_pred, y, x)
                    for key in total_gradient_loss:
                        total_gradient_loss[key] += self.losses.get(key, 0)
                
                for key in total_gradient_loss:
                    total_gradient_loss[key] /= self.batch_size
                
                self.optimizer()

    def evaluate(self, X):
        return np.array(self.forward_propogation(X))
