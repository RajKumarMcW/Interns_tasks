import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split

class KNNClassifier:
    def __init__(self, k=3):
        self.k = k
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        print("Build started...")
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        predictions = [self._predict(x) for x in X]
        return np.array(predictions)

    # def _predict(self, x):
    #     distances = [np.linalg.norm(self._convert_to_numeric(x) - self._convert_to_numeric(x_train)) for x_train in self.X_train]
    #     k_indices = np.argsort(distances)[:self.k]
    #     k_nearest_labels = [self.y_train[i] for i in k_indices]
    #     most_common = np.bincount(k_nearest_labels).argmax()
    #     return most_common

    # def _convert_to_numeric(self, data):
    #     # Convert each element in data to numeric type
    #     return np.array([float(val) for val in data])

    def _predict(self, x):
        distances = [np.linalg.norm(x - x_train) for x_train in self.X_train]
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        most_common = np.bincount(k_nearest_labels).argmax()
        return most_common

#########another one not useful
        # distances = [np.linalg.norm(x - x_train) for x_train in self.X_train]
        # k_indices = np.argsort(distances)[:self.k]
        
        # print("Indices causing the error:", k_indices)
        
        # k_nearest_labels = [self.y_train.iloc[i] for i in k_indices]
        # most_common = np.bincount(k_nearest_labels).argmax()
        # return most_common

# # class Inbuild_KNN:


# import numpy as np
# from sklearn.metrics import accuracy_score, confusion_matrix
# from sklearn.model_selection import train_test_split

# class KNNClassifier:
#     def __init__(self, k=3):
#         self.k = k
#         self.X_train = None
#         self.y_train = None

#     def fit(self, X, y):
#         print("Build started...")
#         self.X_train = X
#         self.y_train = y

#     def predict(self, X):
#         predictions = [self._predict(x) for x in X]
#         return np.array(predictions)

#     def _predict(self, x):
#         x_numeric = self._convert_to_numeric(x).flatten()
#         distances = [np.linalg.norm(x_numeric - x_train) for x_train in self.X_train]
#         k_indices = np.argsort(distances)[:self.k]
#         k_nearest_labels = [self.y_train[i] for i in k_indices]
#         most_common = np.bincount(k_nearest_labels).argmax()
#         return most_common

#     def _convert_to_numeric(self, data):
#         try:
#             # Convert each element in data to numeric type and flatten the array
#             return np.array([float(val) for val in data]).flatten()
#         except ValueError as e:
#             print(f"Error converting values to numeric: {e}")
#             print(f"Problematic data: {data}")
#             return np.array([0.0] * len(data)).flatten()

