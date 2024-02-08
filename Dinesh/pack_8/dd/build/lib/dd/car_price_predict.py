import numpy as np
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

class RandomForestRegressor:
    def __init__(self, n_estimators=100, max_depth=None, min_samples_split=2, min_samples_leaf=1):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.trees = []

    def fit(self, X, y):
        self.trees = []
        for _ in range(self.n_estimators):
            bootstrap_indices = np.random.choice(len(X), len(X), replace=True)
            bootstrap_X = X.iloc[bootstrap_indices].reset_index(drop=True)  
            bootstrap_y = y.iloc[bootstrap_indices].reset_index(drop=True)
            tree = DecisionTreeRegressor(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf
            )
            tree.fit(bootstrap_X, bootstrap_y)
            self.trees.append(tree)

    def predict(self, X):
        predictions = np.zeros((len(X), len(self.trees)))

        for i, tree in enumerate(self.trees):
            predictions[:, i] = tree.predict(X)

        return np.mean(predictions, axis=1)

class DecisionTreeRegressor:
    def __init__(self, max_depth=None, min_samples_split=2, min_samples_leaf=1):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.tree = None

    def fit(self, X, y):
        print("Predicting the accuracy...")
        self.tree = self._build_tree(X, y, depth=0)

    def _build_tree(self, X, y, depth):
        num_samples, num_features = X.shape
        unique_classes = np.unique(y)

        # If only one class in the node or max depth is reached, create a leaf node
        if len(unique_classes) == 1 or (self.max_depth is not None and depth == self.max_depth):
            return {'value': np.mean(y)}

        # Find the best split
        best_split = self._find_best_split(X, y)

        # If no split is found, create a leaf node
        if best_split is None:
            return {'value': np.mean(y)}

        feature_index, threshold = best_split

        # Split the data
        left_mask = X.iloc[:, feature_index] <= threshold
        right_mask = ~left_mask

        # Recursively build the left and right subtrees
        left_subtree = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right_subtree = self._build_tree(X[right_mask], y[right_mask], depth + 1)

        return {
            'feature_index': feature_index,
            'threshold': threshold,
            'left': left_subtree,
            'right': right_subtree
        }

    def _find_best_split(self, X, y):
        num_samples, num_features = X.shape

        if num_samples <= self.min_samples_split:
            return None

        # Calculate the variance of the target values
        current_variance = np.var(y)

        best_split = None
        best_variance_reduction = 0

        for feature_index in range(num_features):
            # Sort the feature values
            feature_values = np.sort(np.unique(X.iloc[:, feature_index]))

            for threshold in feature_values:
                # Split the data
                left_mask = X.iloc[:, feature_index] <= threshold
                right_mask = ~left_mask

                if np.sum(left_mask) < self.min_samples_leaf or np.sum(right_mask) < self.min_samples_leaf:
                    continue

                # Calculate the variance reduction
                left_variance = np.var(y[left_mask])
                right_variance = np.var(y[right_mask])
                variance_reduction = current_variance - (len(y[left_mask]) / len(y) * left_variance +
                                                          len(y[right_mask]) / len(y) * right_variance)

                # Update the best split if needed
                if variance_reduction > best_variance_reduction:
                    best_variance_reduction = variance_reduction
                    best_split = (feature_index, threshold)

        return best_split

    def predict(self, X):
        return np.array([self._predict_tree(x, self.tree) for x in X.values])

    def _predict_tree(self, x, tree):
        if 'value' in tree:
            return tree['value']

        if x[tree['feature_index']] <= tree['threshold']:
            return self._predict_tree(x, tree['left'])
        else:
            return self._predict_tree(x, tree['right'])



