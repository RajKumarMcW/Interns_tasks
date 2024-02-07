import numpy as np

class RidgeRegression:
    def __init__(self, learning_rate=0.01, iterations=1000, lambda_param=1):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.lambda_param = lambda_param
        self.weights = 0
        self.bias = 0

    def fit(self, X, y):
        n_samples, n_features = X.shape

        # Initialize weights and bias
        self.weights = np.zeros((n_features, 1))
        self.bias = 0

        for _ in range(self.iterations):
            # Predictions
            y_pred = np.dot(X, self.weights) + self.bias

            # Calculate gradients
            dw = (2 / n_samples) * (np.dot(X.T, (y_pred - y)) + self.lambda_param * self.weights)
            db = (2 / n_samples) * np.sum(y_pred - y) + 2 * self.lambda_param * self.bias

            # Update weights and bias
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias
