import numpy as np
import xgboost as xgb

class XGBoostClassifier:
    def __init__(self, learning_rate=0.1, num_iterations=100, max_depth=3, num_classes=4):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.max_depth = max_depth
        self.num_classes = num_classes
        self.model = None

    def fit(self, X_train, y_train, sample_weights=None):
        dtrain = xgb.DMatrix(X_train, label=y_train)

        params = {
            'objective': 'multi:softprob',
            'eta': self.learning_rate,
            'max_depth': self.max_depth,
            'num_class': self.num_classes,
            'eval_metric': 'mlogloss'
        }

        # Create a weight array if sample_weights is provided
        if sample_weights is not None:
            weights = np.repeat(sample_weights, np.bincount(y_train))
            dtrain.set_weight(weights)

        # Train the XGBoost model
        self.model = xgb.train(params, dtrain, num_boost_round=self.num_iterations)

    def predict(self, X_test):
        # Convert X_test to a DMatrix
        dtest = xgb.DMatrix(X_test)

        # Make probability predictions for each class
        y_pred_probs = self.model.predict(dtest)

        # Convert probabilities to class predictions
        predictions = np.argmax(y_pred_probs, axis=1)

        return predictions

    def evaluate_metrics(self, true_labels, predicted_labels):
        # Confusion Matrix
        confusion_matrix = np.zeros((self.num_classes, self.num_classes), dtype=int)
        for true_label, predicted_label in zip(true_labels, predicted_labels):
            confusion_matrix[true_label, predicted_label] += 1

        # Precision, Recall, F1 Score, Accuracy for each class
        precision = np.zeros(self.num_classes)
        recall = np.zeros(self.num_classes)
        f1_score = np.zeros(self.num_classes)

        for i in range(self.num_classes):
            true_positive = confusion_matrix[i, i]
            false_positive = np.sum(confusion_matrix[:, i]) - true_positive
            false_negative = np.sum(confusion_matrix[i, :]) - true_positive

            # Precision
            precision[i] = true_positive / (true_positive + false_positive) if (true_positive + false_positive) != 0 else 0

            # Recall
            recall[i] = true_positive / (true_positive + false_negative) if (true_positive + false_negative) != 0 else 0

            # F1 Score
            f1_score[i] = 2 * (precision[i] * recall[i]) / (precision[i] + recall[i]) if (precision[i] + recall[i]) != 0 else 0

        # Accuracy
        accuracy = np.mean(true_labels == predicted_labels)

        # Macro-averaged metrics
        macro_precision = np.mean(precision)
        macro_recall = np.mean(recall)
        macro_f1_score = np.mean(f1_score)

        return accuracy, macro_precision, macro_recall, macro_f1_score
