import numpy as np

class KNeighborsClassifier:
    def __init__(self, n_neighbors=5):
        self.n_neighbors = n_neighbors

    def fit(self, X_train, y_train):
        if isinstance(X_train, np.ndarray):
            self.X_train = X_train
        else:
            self.X_train = X_train.to_numpy()

        if isinstance(y_train, np.ndarray):
            self.y_train = y_train
        else:
            # Convert to numpy array if it's not
            self.y_train = y_train.to_numpy()


    def predict(self, X_test):
        y_pred = []
        if isinstance(X_test, np.ndarray):
            X_test = X_test
        else:
            X_test = X_test.to_numpy()

        for sample in X_test:
            distances = np.linalg.norm(self.X_train - sample, axis=1)
            nearest_neighbors_indices = np.argsort(distances)[:self.n_neighbors]
            nearest_neighbors_labels = self.y_train[nearest_neighbors_indices]
            unique_labels, counts = np.unique(nearest_neighbors_labels, return_counts=True)
            predicted_label = unique_labels[np.argmax(counts)]
            y_pred.append(predicted_label)
        return np.array(y_pred)