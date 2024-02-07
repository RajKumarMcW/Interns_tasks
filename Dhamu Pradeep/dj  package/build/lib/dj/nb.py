import numpy as np
from collections import defaultdict
from scipy.sparse import issparse

class NaiveBayesClassifier:
    def __init__(self):
        self.class_probs = None
        self.word_probs = None
        self.classes = None

    def fit(self, X_train, y_train):
        # Convert X_train to a numpy array
        X_train = X_train.toarray() if issparse(X_train) else np.array(X_train)

        # Convert y_train to integers if it's not already
        y_train = np.array(y_train)

        # Calculate class probabilities
        self.classes, class_counts = np.unique(y_train, return_counts=True)
        self.class_probs = class_counts / len(y_train)

        # Create a dictionary to store word counts for each class
        word_counts_per_class = defaultdict(lambda: np.zeros(X_train.shape[1], dtype=int))

        # Count occurrences of each word in each class
        for i, c in enumerate(self.classes):
            X_c = X_train[y_train == c]
            word_counts_per_class[c] = np.sum(X_c, axis=0)

        # Calculate word probabilities for each class using Laplace smoothing
        self.word_probs = {c: (word_counts.astype(int) + 1) / (np.sum(word_counts) + X_train.shape[1])
                           for c, word_counts in word_counts_per_class.items()}

    def predict(self, X_test):
        predictions = []

        # Convert X_test to a numpy array
        X_test = X_test.toarray() if issparse(X_test) else np.array(X_test)

        for x in X_test:
            # Calculate log probabilities for each class
            log_probs = np.log(self.class_probs)

            # Add log probabilities for each word in the document
            for word_index, count in enumerate(x):
                for c in self.classes:
                    log_probs[c] += count * np.log(self.word_probs[c][word_index])

            # Predict the class with the highest log probability
            prediction = np.argmax(log_probs)
            predictions.append(prediction)

        return predictions

    def evaluate_metrics(self, true_labels, predicted_labels):
        # Confusion Matrix
        confusion_matrix = np.zeros((len(self.classes), len(self.classes)), dtype=int)
        for true_label, predicted_label in zip(true_labels, predicted_labels):
            confusion_matrix[true_label, predicted_label] += 1

        # Precision, Recall, F1 Score
        precision = np.zeros(len(self.classes))
        recall = np.zeros(len(self.classes))
        f1 = np.zeros(len(self.classes))

        for i in range(len(self.classes)):
            true_positive = confusion_matrix[i, i]
            false_positive = np.sum(confusion_matrix[:, i]) - true_positive
            false_negative = np.sum(confusion_matrix[i, :]) - true_positive

            # Precision
            precision[i] = true_positive / (true_positive + false_positive) if (true_positive + false_positive) != 0 else 0

            # Recall
            recall[i] = true_positive / (true_positive + false_negative) if (true_positive + false_negative) != 0 else 0

            # F1 Score
            f1[i] = 2 * (precision[i] * recall[i]) / (precision[i] + recall[i]) if (precision[i] + recall[i]) != 0 else 0

        # Calculate the average precision, recall, and F1 Score
        avg_precision = np.mean(precision)
        avg_recall = np.mean(recall)
        avg_f1 = np.mean(f1)

        # Accuracy
        accuracy = np.mean(true_labels == predicted_labels)

        return accuracy, avg_precision, avg_recall, avg_f1
