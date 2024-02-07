import numpy as np

def accuracy(y_true, y_pred):
    correct = np.sum(y_true == y_pred)
    total = len(y_true)
    return correct / total

def precision(y_true, y_pred, class_label=1):
    true_positives = np.sum((y_true == class_label) & (y_pred == class_label))
    predicted_positives = np.sum(y_pred == class_label)
    return true_positives / predicted_positives

def recall(y_true, y_pred, class_label=1):
    true_positives = np.sum((y_true == class_label) & (y_pred == class_label))
    actual_positives = np.sum(y_true == class_label)
    return true_positives / actual_positives

def f1_score(y_true, y_pred, class_label=1):
    prec = precision(y_true, y_pred, class_label)
    rec = recall(y_true, y_pred, class_label)
    return 2 * (prec * rec) / (prec + rec)

def confusion_matrix_custom(y_true, y_pred):
    unique_labels = np.unique(np.concatenate([y_true, y_pred]))
    matrix = np.zeros((len(unique_labels), len(unique_labels)), dtype=int)
    for i, true_label in enumerate(unique_labels):
        for j, pred_label in enumerate(unique_labels):
            matrix[i, j] = np.sum((y_true == true_label) & (y_pred == pred_label))
    return matrix

def classification_metric(y_test, y_pred):
    y_test=y_test.to_numpy()
    print(f'''
Accuracy :{accuracy(y_test, y_pred):.2f}
Classication Report:
Precision(class 0): {precision(y_test, y_pred, class_label=0):.2f}
Recall(class 0)   : {recall(y_test, y_pred, class_label=0):.2f}
F1-score(class 0) : {f1_score(y_test, y_pred, class_label=0):.2f}

Precision(class 1): {precision(y_test, y_pred, class_label=1):.2f}
Recall(class 1)   : {recall(y_test, y_pred, class_label=1):.2f}
F1-score(class 1) : {f1_score(y_test, y_pred, class_label=1):.2f}

Confusion Matrix:
{confusion_matrix_custom(y_test, y_pred)}
''')


def cmean_absolute_error(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

def cmean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def root_mean_squared_error(y_true, y_pred):
    return np.sqrt(cmean_squared_error(y_true, y_pred))

def r_squared(y_true, y_pred):
    y_mean = np.mean(y_true)
    ss_total = np.sum((y_true - y_mean) ** 2)
    ss_residual = np.sum((y_true - y_pred) ** 2)
    return 1 - (ss_residual / ss_total)

def regression_metric(y_test, y_pred):
    y_test=y_test.to_numpy()
    print(f'''
Mean Absolute Error (MAE)     : {cmean_absolute_error(y_test, y_pred):.2f}
Mean Squared Error (MSE)      : {cmean_squared_error(y_test, y_pred):.2f}
Root Mean Squared Error (RMSE): {root_mean_squared_error(y_test, y_pred):.2f}
R-squared (R2) Score          : {r_squared(y_test, y_pred):.2f}
''')