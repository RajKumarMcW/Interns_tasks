import math
import numpy as np

class DecisionStump:
    def __init__(self):
        self.polarity = 1
        self.feature_index = None
        self.threshold = None
        self.alpha = None

class Adaptive_Boost:

    def __init__(self, n_clf=5):
        self.n_clf = n_clf

    def __str__(self):
        return "Adaptive_Boost"

    def cfit(self, X, y):
        X = X.to_numpy()
        y = np.where(y.to_numpy() == 0, -1, 1)
        n_samples, n_features = X.shape

        # Initialize weights to 1/N
        w = np.full(n_samples, 1 / n_samples)

        self.clfs = []

        for _ in range(self.n_clf):
            clf = DecisionStump()
            min_error = float('inf')

            for feature_i in range(n_features):
                feature_values = np.expand_dims(X[:, feature_i], axis=1)
                unique_values = np.unique(feature_values)

                for threshold in unique_values:
                    p = 1
                    prediction = np.ones(n_samples)
                    negative_idx = (feature_values < threshold).flatten()
                    prediction[negative_idx] = -1
                    error = np.sum(w[y != prediction])

                    if error > 0.5:
                        error = 1 - error
                        p = -1

                    if error < min_error:
                        clf.polarity = p
                        clf.threshold = threshold
                        clf.feature_index = feature_i
                        min_error = error

            clf.alpha = 0.5 * math.log((1.0 - min_error) / (min_error + 1e-10))

            predictions = np.ones(n_samples)
            negative_idx = (clf.polarity * X[:, clf.feature_index] < clf.polarity * clf.threshold)
            predictions[negative_idx] = -1

            w *= np.exp(-clf.alpha * y * predictions)
            w /= np.sum(w)

            self.clfs.append(clf)

    def cpredict(self, X):
        X = X.to_numpy()
        n_samples = X.shape[0]
        y_pred = np.zeros(n_samples)

        for clf in self.clfs:
            predictions = np.ones(n_samples)
            negative_idx = (clf.polarity * X[:, clf.feature_index] < clf.polarity * clf.threshold)
            predictions[negative_idx] = -1
            y_pred += clf.alpha * predictions

        y_pred = np.sign(y_pred).flatten()
        y_pred[y_pred == -1] = 0
        return y_pred

