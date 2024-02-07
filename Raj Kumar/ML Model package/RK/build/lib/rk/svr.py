import numpy as np

class Support_Vector_Regressor:
    def __init__(self):
        # Store Necessary Data Used Across All Methods #
        self.trainX = []
        self.bias = []
        self.alpha = []
        self.modelNumber = 0

    def __str__(self):
        return f"Support_Vector_Regressor"

    def computeGram(self, x, y, penalize='L1'):
        x_values = x.values
        y_values = y.values
        K = np.dot(x_values, y_values.T)

        if penalize == 'L1':
            return K
        elif penalize == 'L2':
            return K + np.eye(len(K), len(K)) 

    def cfit(self, trainX, trainZ):
        self.trainX = trainX
        psi = self.computeGram(self.trainX, self.trainX, penalize='L2')

        A = np.vstack((np.append(0, np.ones((1, len(self.trainX)))), np.hstack((np.ones((len(self.trainX), 1)), psi))))
        b = np.append(0, trainZ).reshape(-1, 1)
        x_sol = np.linalg.solve(A, b)
        self.bias = x_sol[0]
        self.alpha = x_sol[1:]

    def cpredict(self, testX):
        predPsi = self.computeGram(testX, self.trainX, penalize='L1')
        predicted = np.dot(predPsi, self.alpha.reshape(-1, 1)) + self.bias
        return predicted