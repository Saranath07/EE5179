import numpy as np

class ParameterInitializer:
    def __init__(self, initialization='random'):
        """
        Parameters:
        initialization (str): 'random' for uniform random initialization, 'gaussian' for Gaussian distribution.
        """
        self.initialization = initialization

    def initialize_parameters(self, inputs, hidden_layers, outputs):
        """
        Initializes the parameters for a neural network.

        Args:
            inputs (int): The number of input nodes.
            hidden_layers (list): A list of integers representing the number of nodes in each hidden layer.
            outputs (int): The number of output nodes.

        Returns:
            dict: A dictionary containing the initialized parameters, with keys "W1" and "b1" for the first layer, and "W{i+1}" and "b{i+1}" for each hidden layer, where i is the index of the layer. The last two keys are "W{len(hidden_layers) + 1}" and "b{len(hidden_layers) + 1}" for the output layer.

        Raises:
            None.

        Examples:
            >>> initialize_parameters(2, [3, 2], 1)
            {'W1': array([[0.12345678, 0.23456789],
                           [0.34567891, 0.45678901]]), 'b1': array([0.56789012, 0.67890123]), 'W2': array([[0.78901234, 0.89012345],
                           [0.90123456, 0.01234567]]), 'b2': array([0.34567891, 0.45678901]), 'W3': array([[0.67890123, 0.78901234]]), 'b3': array([0.45678901])}
        """
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
