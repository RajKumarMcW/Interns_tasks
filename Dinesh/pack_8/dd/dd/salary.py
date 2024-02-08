import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score
class AdaBoostClassifierCustom:
    def __init__(self, n_estimators=100, random_state=None):
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.models = []
        self.alphas = []

    def fit(self, X, y):
        np.random.seed(self.random_state)
        m, n = X.shape
        classes = np.unique(y)
        num_classes = len(classes)
        weights = np.ones((m, num_classes)) / m
        self.models = []
        self.alphas = []

        for _ in range(self.n_estimators):
            model = DecisionStump()
            model.fit(X, y, weights)

            # Make predictions
            predictions = model.predict(X)

            # Calculate weighted error
            err = np.sum(weights * (predictions != y[:, None]))

            # Calculate model weight (alpha)
            alpha = 0.5 * np.log((1 - err) / max(err, 1e-10))
            self.models.append((model, alpha))

            # Update sample weights
            weights *= np.exp(-alpha * y[:, None] * predictions)
            weights /= np.sum(weights, axis=1)[:, None]

            self.alphas.append(alpha)

    def predict(self, X):
        # Combine predictions from all weak learners
        predictions = np.zeros((X.shape[0], len(self.models[0][0].classes)))
        for model, alpha in self.models:
            predictions += alpha * model.predict(X)

        return np.argmax(predictions, axis=1)
        


class DecisionStump:
    def __init__(self):
        self.feature_index = None
        self.threshold = None
        self.polarity = None
        self.classes = None

    def fit(self, X, y, weights):
        print("Build started...")
        m, n = X.shape
        self.classes = np.unique(y)

        # Initialize to a large value
        min_error = m
        best_feature = None
        best_threshold = None
        best_polarity = None

        for feature in range(n):
            # Sort the feature values
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                for polarity in [1, -1]:
                    predictions = np.ones((m, len(self.classes)))
                    predictions[polarity * X[:, feature] < polarity * threshold, :] = -1

                    # Calculate weighted error
                    err = np.sum(weights * (predictions != y[:, None]))

                    # Update the decision stump if this threshold gives a lower error
                    if err < min_error:
                        min_error = err
                        best_feature = feature
                        best_threshold = threshold
                        best_polarity = polarity

        # Update the decision stump if there was an improvement
        if best_feature is not None:
            self.feature_index = best_feature
            self.threshold = best_threshold
            self.polarity = best_polarity

    def predict(self, X):
        # Make predictions based on the decision stump
        predictions = np.ones((X.shape[0], len(self.classes)))
        if self.feature_index is not None:
            predictions[X[:, self.feature_index] < self.threshold, :] = -1
        return predictions



