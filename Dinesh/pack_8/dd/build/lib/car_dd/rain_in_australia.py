import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Function to add a bias term (column of 1s) to the feature matrix
def add_bias_term(X):
    return np.c_[np.ones(X.shape[0]), X]

# Sigmoid activation function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Cost function for logistic regression
def cost_function(X, y, weights):
    m = len(y)
    predictions = sigmoid(np.dot(X, weights))
    cost = (-1/m) * np.sum(y * np.log(predictions) + (1-y) * np.log(1-predictions))
    return cost

# Gradient descent for logistic regression
def gradient_descent(X, y, weights, learning_rate, iterations):
    m = len(y)
    costs = []
    for _ in range(iterations):
        predictions = sigmoid(np.dot(X, weights))
        gradient = np.dot(X.T, (predictions - y)) / m
        weights -= learning_rate * gradient
        cost = cost_function(X, y, weights)
        costs.append(cost)
    return weights, costs

# Predict function
def predict(X, weights):
    predictions = sigmoid(np.dot(X, weights))
    return [1 if p >= 0.5 else 0 for p in predictions]

# Assuming 'Yes' is the positive class and 'No' is the negative class
y_train_labels = train_targets.replace({'Yes': 1, 'No': 0}).values

# Training set
X_train = add_bias_term(train_inputs_encoded[encoded_cols].values)
y_train = y_train_labels

# Ensure encoded_cols is a subset of columns in val_inputs
encoded_cols = [col for col in encoded_cols if col in val_inputs.columns]

# Validation set
X_val = add_bias_term(val_inputs[encoded_cols].values)
y_val = val_targets.values

# Initialize weights
initial_weights = np.zeros(X_train.shape[1])

# Set hyperparameters
learning_rate = 0.01
iterations = 1000

# Train the model
trained_weights, costs = gradient_descent(X_train, y_train, initial_weights, learning_rate, iterations)
# Assuming 'Yes' is the positive class and 'No' is the negative class
y_train_labels = train_targets.replace({'Yes': 1, 'No': 0}).values

# Training set
X_train = add_bias_term(train_inputs_encoded[encoded_cols].values)
y_train = y_train_labels

# Validation set
y_val_labels = val_targets.replace({'Yes': 1, 'No': 0}).values
X_val = add_bias_term(val_inputs[encoded_cols].values)
y_val = y_val_labels

# Initialize weights
initial_weights = np.zeros(X_train.shape[1])

# Set hyperparameters
learning_rate = 0.01
iterations = 1000

# Train the model
trained_weights, costs = gradient_descent(X_train, y_train, initial_weights, learning_rate, iterations)

# Plot the cost over iterations to check for convergence
plt.plot(costs)
plt.xlabel('Iterations')
plt.ylabel('Cost')
plt.title('Cost over Iterations')
plt.show()

# Validate the model
val_predictions = predict(X_val, trained_weights)
val_accuracy = np.mean(val_predictions == y_val)
print(f"Validation Accuracy: {val_accuracy}")

# Test set
X_test = add_bias_term(test_inputs[encoded_cols].values)
y_test_labels = test_targets.replace({'Yes': 1, 'No': 0}).values
y_test = y_test_labels

# Make predictions on the test set
test_predictions = predict(X_test, trained_weights)

# Calculate accuracy on the test set
test_accuracy = np.mean(test_predictions == y_test)
print(f"Test Accuracy: {test_accuracy}")
