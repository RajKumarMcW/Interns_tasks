import numpy as np

class Linear_Regression:
    def __init__(self, learning_rate=0.00001, epsilon=0.9):
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.weights = None

    def __str__(self):
        return f"Linear_Regression"

    def cfit(self, x, y):
        # Step 1: Insert a new column with ones for y-intercept
        x = x.to_numpy()
        y = y.to_numpy()

        regression = np.c_[x, np.ones(len(x))]

        # Step 2: Declare the weights with the same width as x
        self.weights = np.ones(regression.shape[1])

        # Step 3: Implement gradient descent
        norma = 1
        while norma > self.epsilon:
            # Step 3.1: Compute the partial derivative
            y_pred = regression @ self.weights.T
            partial = regression.T @ (y - y_pred)

            # Step 3.2: Compute the norma
            norma = np.sum(np.sqrt(np.square(partial)))

            # Step 3.3: Adjust the weights
            self.weights = self.weights.T + (self.learning_rate * partial)

            # Check for divergence
            if np.isnan(norma):
                print('The model diverged. Try using a smaller learning rate.')

    def cpredict(self, x):
        x=x.to_numpy()
        if self.weights is None:
            raise ValueError("Model has not been trained. Call fit() first.")

        # Predict using the learned weights
        regression_input = np.c_[x, np.ones(len(x))]
        return regression_input @ self.weights.T
    