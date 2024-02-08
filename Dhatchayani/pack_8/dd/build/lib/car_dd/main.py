# main.py

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from knn_classifier import KNNClassifier 
from decision_tree import DecisionTree
from random_forest_regressor import RandomForestRegressor
from preprocessing import preprocess_data

# Load your data (replace 'your_dataset.csv' with your actual dataset)
df_avocado = pd.read_csv('your_dataset.csv')

# Example usage for KNNClassifier
# Assuming 'target_column' is the column to be predicted
knn_target_column = 'target_column'
X_train_knn, X_test_knn, y_train_knn, y_test_knn = preprocess_data(df_avocado, knn_target_column)
knn_custom_classifier = KNNClassifier(k=3)
knn_custom_classifier.fit(X_train_knn, y_train_knn)
accuracy_knn, conf_matrix_knn = knn_custom_classifier.evaluate(X_test_knn, y_test_knn)

# Example usage for DecisionTree
# Assuming 'target_column' is the column to be predicted
tree_target_column = 'target_column'
X_train_tree, X_test_tree, y_train_tree, y_test_tree = preprocess_data(df_avocado, tree_target_column)
tree_model = DecisionTree(max_depth=None)
tree_model.fit(X_train_tree, y_train_tree)
y_pred_tree = tree_model.predict(X_test_tree)

# Example usage for RandomForestRegressor
# Assuming 'target_column' is the column to be predicted
rf_target_column = 'target_column'
X_train_rf, X_test_rf, y_train_rf, y_test_rf = preprocess_data(df_avocado, rf_target_column)
rf_regressor = RandomForestRegressor(n_estimators=100, max_depth=5, min_samples_split=2, min_samples_leaf=1)
rf_regressor.fit(X_train_rf, y_train_rf)
predictions_rf = rf_regressor.predict(X_test_rf)
