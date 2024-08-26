import numpy as np
from scipy.special import expit

class Activations:

    class ReLU:

        def __call__(self, x):
            """
            Applies the ReLU activation function to the given input x.

            Args:
                x (float): The input value.

            Returns:
                float: The output of the ReLU activation function, which is max(0, x).
            """
            return max(0, x)


        def gradient(self, x):
            """
            Computes the gradient of the ReLU activation function.

            Args:
                x (float): The input value.

            Returns:
                float: The gradient of the ReLU activation function, which is 0 for x <= 0 and 1 for x > 0.
            """
            if x <= 0:
                return 0
            else:
                return 1
    

    class Sigmoid:

        def __call__(self, x):
            """
            Applies the Sigmoid activation function to the given input x.

            Args:
                x (float): The input value.

            Returns:
                float: The output of the Sigmoid activation function.
            """
            return expit(x)

        def gradient(self, x):
            """
            Computes the gradient of the Sigmoid activation function.

            Args:
                x (float): The input value.

            Returns:
                float: The gradient of the Sigmoid activation function.
            """
            return self.__call__(x) * (1 - self.__call__(x))
    



    class Softmax:

        def __call__(self, x):
            """
            Applies the Softmax activation function to the given input x.

            Args:
                x (numpy.ndarray): The input array.

            Returns:
                numpy.ndarray: The output of the Softmax activation function.
            """
            exp_x = np.exp(x - np.max(x))  # Subtracting max(x) for numerical stability
            return exp_x / np.sum(exp_x, axis=0)

        def gradient(self, x):
            """
            Computes the gradient of the Softmax activation function.

            Args:
                x (numpy.ndarray): The input array.

            Returns:
                numpy.ndarray: The Jacobian matrix of the Softmax activation function.
            """
            s = self.__call__(x)
            jacobian_matrix = np.diagflat(s) - np.outer(s, s)
            return jacobian_matrix
        
        def __repr__(self):
            return "Softmax"

    
    class Tanh:

        def __call__(self, x):
            """
            Applies the Tanh activation function to the given input x.

            Args:
                x (numpy.ndarray): The input array.

            Returns:
                numpy.ndarray: The output of the Tanh activation function.
            """
            return np.tanh(x)

        def gradient(self, x):
            """
            Computes the gradient of the Tanh activation function.

            Args:
                x (numpy.ndarray): The input array.

            Returns:
                numpy.ndarray: The derivative of the Tanh activation function.
            """
            return 1 - np.power(self.__call__(x), 2)
    
    class Linear:

        def __call__(self, x):
            """
            Returns the input value.

            Parameters:
                x (Any): The input value.

            Returns:
                Any: The input value.
            """
            return x

        def gradient(self, x):
            """
            Computes the gradient of the Linear activation function.

            Args:
                x (Any): The input value.

            Returns:
                int: The derivative of the Linear activation function, which is always 1.
            """
            return 1