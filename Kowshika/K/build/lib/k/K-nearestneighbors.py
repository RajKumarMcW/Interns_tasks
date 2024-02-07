import numpy as np
from scipy.stats import mode

class K_Nearest_Neighbors_Classifier:

    def __init__(self, K=3):
        self.K = K

    def __str__(self):
        return f"K_Nearest_Neighbors_Classifier"

    # Function to store training set
    def cfit(self, X_train, Y_train):
        self.X_train = X_train.to_numpy()
        self.Y_train = Y_train.to_numpy()

    # Function for prediction
    def cpredict(self, X_test):
        self.X_test = X_test.to_numpy()
        # no_of_test_examples, no_of_features
        m_test, _ = X_test.shape
        # initialize Y_predict
        Y_predict = np.zeros(m_test)
        for i in range(m_test):
            x = self.X_test[i]
            # find the K nearest neighbors from current test example
            neighbors = self.find_neighbors(x)
            # most frequent class in K neighbors
            Y_predict[i] = mode(neighbors)[0]
        return Y_predict

    # Function to find the K nearest neighbors to current test example
    def find_neighbors(self, x):
        # calculate all the euclidean distances between current test example x and training set X_train
        euclidean_distances = np.linalg.norm(self.X_train - x, axis=1)
        # sort Y_train according to euclidean_distance_array and store into Y_train_sorted
        inds = euclidean_distances.argsort()
        Y_train_sorted = self.Y_train[inds]
        return Y_train_sorted[:self.K]
