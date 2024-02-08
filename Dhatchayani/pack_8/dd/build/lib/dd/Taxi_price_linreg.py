import numpy as np

class LinearRegression:
    def __init__(self):
        self.coef_ = None
        self.intercept_ = None

    def fit(self, X, y):
        # Add a column of ones to X for the intercept term
        X = np.c_[np.ones(X.shape[0]), X]

        # Use the normal equation to calculate coefficients
        # theta = (X^T * X)^(-1) * X^T * y
        X_transpose = np.transpose(X)
        self.coefficients = np.linalg.inv(X_transpose.dot(X)).dot(X_transpose).dot(y)

        # Extract coefficients
        self.intercept_ = self.coefficients[0]
        self.coef_ = self.coefficients[1:]

    def predict(self, X):
        # Add a column of ones to X for the intercept term
        X = np.c_[np.ones(X.shape[0]), X]

        # Use the learned coefficients to make predictions
        return X.dot(self.coefficients)

# Example usage:
# Assuming X_train and y_train are your training data
# Assuming X_test is your testing data

# Create and fit the model
# LR = LinearRegression()
# LR.fit(X_train, y_train)

# # Make predictions on the test set
# y_custom_pred = LR.predict(X_test)

# # Print the learned coefficients
# print("Intercept:", LR.intercept_)
# print("Coefficients:", LR.coef_)