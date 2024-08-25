import numpy as np

class ParameterInitializer:
    def __init__(self, initialization='random'):
        """
        Parameters:
        initialization (str): 'random' for uniform random initialization, 'gaussian' for Gaussian distribution.
        """
        self.initialization = initialization

    def initialize_parameters(self, inputs, hidden_layers, outputs):
        parameters = {}

        if self.initialization == 'random':
            parameters["W1"] = np.random.rand(hidden_layers[0], inputs)
            parameters["b1"] = np.random.rand(hidden_layers[0])
            
        elif self.initialization == 'gaussian':
            parameters["W1"] = np.random.randn(hidden_layers[0], inputs)
            parameters["b1"] = np.random.randn(hidden_layers[0])
            for i in range(1, len(hidden_layers)):
                parameters[f"W{i+1}"] = np.random.rand(hidden_layers[i], hidden_layers[i - 1])
                parameters[f"b{i+1}"] = np.random.rand(hidden_layers[i])
            parameters[f"W{len(hidden_layers) + 1}"] = np.random.rand(outputs, hidden_layers[-1])
            parameters[f"b{len(hidden_layers) + 1}"] = np.random.rand(outputs)
        
        return parameters
