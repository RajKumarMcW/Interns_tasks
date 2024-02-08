import numpy as np

class LogisticRegressionCustom:
    def __init__(self, learning_rate=0.01, num_iterations=1000, verbose=False):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.verbose = verbose
        self.weights = None
        self.bias = None

    def sigmoid(self, z):
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z.astype(float)))

    def initialize_parameters(self, num_features):
        self.weights = np.zeros(num_features)
        self.bias = 0

    def fit(self, X, y):
        print("Build started...")
        num_samples, num_features = X.shape
        self.initialize_parameters(num_features)

        for iteration in range(self.num_iterations):
            # Forward pass
            z = np.dot(X, self.weights) + self.bias
            predictions = self.sigmoid(z)

            # Compute gradients
            dz = predictions - y
            dw = np.dot(X.T, dz) / num_samples
            db = np.sum(dz) / num_samples

            # Update parameters
            self.weights = self.weights.astype(float) - self.learning_rate * dw.astype(float)
            # self.bias = self.bias.astype(float) - self.learning_rate * db.astype(float)
            self.bias -= self.learning_rate * db

            # Print loss for every 100 iterations
            if self.verbose and iteration % 100 == 0:
                loss = -(1/num_samples) * np.sum(y * np.log(predictions) + (1 - y) * np.log(1 - predictions))
                print(f"Iteration {iteration}, Loss: {loss}")


    def predict(self, X):
        print("Prediction started...")
        z = np.dot(X, self.weights) + self.bias
        predictions = self.sigmoid(z)
        return np.round(predictions)
