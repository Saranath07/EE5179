import numpy as np
from scipy.special import expit

class Activations:

    class ReLU:

        def __call__(self, x):
            return max(0, x)


        def gradient(self, x):
            if x <= 0:
                return 0
            else:
                return 1
    

    class Sigmoid:

        def __call__(self, x):
            return expit(x)

        def gradient(self, x):
            return self.__call__(x) * (1 - self.__call__(x))
    



    class Softmax:

        def __call__(self, x):
            exp_x = np.exp(x - np.max(x))  # Subtracting max(x) for numerical stability
            return exp_x / np.sum(exp_x, axis=0)

        def gradient(self, x):
            s = self.__call__(x)
            jacobian_matrix = np.diagflat(s) - np.outer(s, s)
            return jacobian_matrix
        
        def __repr__(self):
            return "Softmax"

    
    class Tanh:

        def __call__(self, x):
            return np.tanh(x)

        def gradient(self, x):
            return 1 - np.power(self.__call__(x), 2)
    
    class Linear:

        def __call__(self, x):
            return x

        def gradient(self, x):
            return 1