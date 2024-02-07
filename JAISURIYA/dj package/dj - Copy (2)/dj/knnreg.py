import numpy as np

class KNNRegressor:
    def __init__(self, K=3):
        self.K = K
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        predictions = [self._predict(x) for x in X]
        return np.array(predictions)

    def _predict(self, x):
        # Compute distances between x and all examples in the training set
        distances = [np.sqrt(np.sum((x_train - x) ** 2)) for x_train in self.X_train]
        # Sort by distance and return indices of the first K neighbors
        k_indices = np.argsort(distances)[:self.K]
        # Extract the labels of the K nearest neighbor training samples
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        # Return the mean of K nearest labels
        return np.mean(k_nearest_labels)