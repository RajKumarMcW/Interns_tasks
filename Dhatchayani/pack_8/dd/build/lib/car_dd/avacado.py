import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

class DecisionTree:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth
        self.tree = None

    def mean_squared_error(self, y):
        return np.mean((y - np.mean(y))**2)

    def find_best_split(self, X, y):
        m, n = X.shape
        if m <= 1:
            return None

        total_mean = np.mean(y)
        best_mse = float('inf')
        best_feature = None
        best_threshold = None

        for feature in range(n):
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                left_indices = X[:, feature] < threshold
                right_indices = ~left_indices

                if np.sum(left_indices) > 0 and np.sum(right_indices) > 0:
                    left_mse = self.mean_squared_error(y[left_indices])
                    right_mse = self.mean_squared_error(y[right_indices])
                    mse = (np.sum(left_indices) * left_mse + np.sum(right_indices) * right_mse) / m

                    if mse < best_mse:
                        best_mse = mse
                        best_feature = feature
                        best_threshold = threshold

        return best_feature, best_threshold

    def build_tree(self, X, y, depth=1):
        if self.max_depth is not None and depth == self.max_depth or self.mean_squared_error(y) == 0:
            return Node(value=np.mean(y))

        feature, threshold = self.find_best_split(X, y)

        if feature is None:
            return Node(value=np.mean(y))

        left_indices = X[:, feature] < threshold
        right_indices = ~left_indices

        left_subtree = self.build_tree(X[left_indices], y[left_indices], depth + 1)
        right_subtree = self.build_tree(X[right_indices], y[right_indices], depth + 1)

        return Node(feature=feature, threshold=threshold, left=left_subtree, right=right_subtree)

    def fit(self, X, y):
        self.tree = self.build_tree(X, y)

    def predict_tree(self, node, X):
        if node.value is not None:
            return node.value

        if X[node.feature] < node.threshold:
            return self.predict_tree(node.left, X)
        else:
            return self.predict_tree(node.right, X)

    def predict(self, X):
        return np.array([self.predict_tree(self.tree, x) for x in X])

# Assuming X_train_scaled, y_train, X_test_scaled, y_test are defined
# Instantiate the DecisionTree class
tree_model = DecisionTree(max_depth=None)

# Fit the model on the training data
tree_model.fit(X_train_scaled, y_train.to_numpy())

# Make predictions on the test set
y_pred_tree = tree_model.predict(X_test_scaled)

# Calculate Mean Squared Error on the test set
mse_tree = mean_squared_error(y_test.to_numpy(), y_pred_tree)
print('Mean Squared Error on Test Set:', mse_tree)

# Calculate R-squared error on the test set
r2_tree = r2_score(y_test, y_pred_tree)
print("R-squared error: ", r2_tree)
