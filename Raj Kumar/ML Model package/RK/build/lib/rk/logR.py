import numpy as np

class Logistic_Regression:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None

    def __str__(self):
        return f"Logistic_Regression"
    
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    

    def cfit(self, X, y):
        # Initialize weights and bias
        self.weights = np.zeros(X.shape[1])
        self.bias = 0

        for _ in range(self.n_iterations):
            # Linear combination of features and weights
            linear_model = np.dot(X, self.weights) + self.bias

            # Apply the sigmoid function to get probabilities
            y_pred = self.sigmoid(linear_model)

            # Calculate the gradient
            dw = (1 / len(y)) * np.dot(X.T, (y_pred - y))
            db = (1 / len(y)) * np.sum(y_pred - y)

            # Update weights and bias
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def cpredict(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        probabilities = self.sigmoid(linear_model)
        predictions = np.round(probabilities)
        return predictions