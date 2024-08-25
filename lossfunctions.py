import numpy as np

class LossFunctions:

    class SSE:
        def __init__(self):
            pass

        def __call__(self, y_pred, y_true):
            """
            Computes the Mean Squared Error (MSE) between predicted and true values.

            Parameters:
            y_pred (ndarray): The predicted values.
            y_true (ndarray): The true values.

            Returns:
            float: The mean squared error.
            """
            return np.sum(np.square(y_pred - y_true))

        def gradient(self, y_pred, y_true):
            """
            Computes the gradient of the Mean Squared Error (MSE) Loss with respect to y_pred.
            
            Parameters:
            y_pred (ndarray): The predicted values.
            y_true (ndarray): The true values.
            
            Returns:
            ndarray: The gradient with respect to y_pred.
            """
            return 2 * (y_pred - y_true)
        
        def __repr__(self):
            return "SSE"


    class CrossEntropy:
        def __init__(self):
            pass

        def __call__(self, y_pred, y_true):
            """
            Compute the Cross-Entropy Loss.
            
            Parameters:
            y_pred (ndarray): The predicted probabilities (output of softmax) of shape (num_classes,).
            y_true (ndarray): The one-hot encoded true labels of shape (num_classes,).
            
            Returns:
            float: The cross-entropy loss.
            """
            # Clip y_pred to prevent log(0) errors
            y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
            return -np.sum(y_true * np.log(y_pred))

        def gradient(self, y_pred, y_true):
            """
            Compute the gradient of the Cross-Entropy Loss with respect to y_pred.
            
            Parameters:
            y_pred (ndarray): The predicted probabilities (output of softmax) of shape (num_classes,).
            y_true (ndarray): The one-hot encoded true labels of shape (num_classes,).
            
            Returns:
            ndarray: The gradient with respect to y_pred, of shape (num_classes,).
            """
            # Clip y_pred to prevent division by zero
            y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
            return - (y_true / y_pred)
        
        def __repr__(self):
            return "CrossEntropy"
