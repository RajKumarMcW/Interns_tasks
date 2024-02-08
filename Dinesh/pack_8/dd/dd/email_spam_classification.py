import numpy as np

class MultinomialNaiveBayes:
    def __init__(self, alpha=1.0):
        self.alpha = alpha  # Laplace smoothing parameter
        self.class_probs = None
        self.feature_probs = None
        self.classes = None

    def fit(self, X, y):
        # Calculate class probabilities
        print("Build started...")
        self.classes, class_counts = np.unique(y, return_counts=True)
        self.class_probs = class_counts / len(y)

        # Calculate feature probabilities
        self.feature_probs = {}
        for c in self.classes:
            class_mask = (y == c)
            class_features = X[class_mask]
            total_class_instances = len(class_features)
            feature_probs = (np.sum(class_features, axis=0) + self.alpha) / (total_class_instances + self.alpha * X.shape[1])
            self.feature_probs[c] = feature_probs

    def predict(self, X):
        predictions = [self.predict_instance(xi) for xi in X]
        return np.array(predictions)

    # def predict_instance(self, x):
    #     class_scores = []
    #     for c in self.classes:
    #         class_score = np.log(self.class_probs[c])
    #         class_score += np.sum(np.log(self.feature_probs[c]) * x)
    #         class_scores.append(class_score)
    #     return self.classes[np.argmax(class_scores)]
    def predict_instance(self, x):
        class_scores = []
        for c in self.classes:
            class_score = np.log(self.class_probs[c])
            class_score += np.sum((self.feature_probs[c] + 1e-10) ** x)
            class_scores.append(class_score)
        return self.classes[np.argmax(class_scores)]


