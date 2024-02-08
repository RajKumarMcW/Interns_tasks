import numpy as np
from sklearn.preprocessing import StandardScaler

class SVM:
    def _init_(self, C=1.0, tol=1e-2, max_iter=100, kernel='linear'):
        self.C = C
        self.tol = tol
        self.max_iter = max_iter
        self.kernel = kernel

    def fit(self, X, y):
        self.X = X
        self.y = y
        self.m, self.n = X.shape

        # Initialize parameters
        self.alpha = np.zeros(self.m)
        self.b = 0.0
        self.E = self.decision_function(X) - y

        # Train the SVM
        for _ in range(self.max_iter):
            changed_alphas = 0
            for i in range(self.m):
                if (y[i] * self.E[i] < -self.tol and self.alpha[i] < self.C) or \
                   (y[i] * self.E[i] > self.tol and self.alpha[i] > 0):
                    j = self.random_index(i)
                    if self.take_step(i, j):
                        changed_alphas += 1

            if changed_alphas == 0:
                break

        # Compute the bias term
        self.b = self.calculate_b()

    def decision_function(self, X):
        assert X.shape[1] == self.X.shape[1], "Number of features in X must match the training data."
        assert self.alpha.ndim == 1, "self.alpha should be a 1D array."
        assert self.y.ndim == 1, "self.y should be a 1D array."

        decision = np.dot(X, self.X.T)
        decision = np.sum(decision * self.alpha * self.y) + self.b
        return decision

    def random_index(self, i):
        j = i
        while j == i:
            j = np.random.randint(self.m)
        return j

    def take_step(self, i, j):
        if i == j:
            return False

        # Compute bounds L and H
        L, H = self.compute_L_H(i, j)

        if L == H:
            return False

        # Compute eta
        eta = 2.0 * self.X[i] @ self.X[j].T - self.X[i] @ self.X[i].T - self.X[j] @ self.X[j].T

        if eta >= 0:
            return False

        # Compute new alpha_j
        new_alpha_j = self.alpha[j] - self.y[j] * (self.E[i] - self.E[j]) / eta

        # Clip new_alpha_j
        new_alpha_j = self.clip_alpha(new_alpha_j, L, H)

        if abs(self.alpha[j] - new_alpha_j) < 1e-5:
            return False

        # Update alpha_i
        new_alpha_i = self.alpha[i] + self.y[i] * self.y[j] * (self.alpha[j] - new_alpha_j)

        # Update bias terms
        new_b1 = self.b - self.E[i] - self.y[i] * (new_alpha_i - self.alpha[i]) * self.X[i] @ self.X[i].T \
                 - self.y[j] * (new_alpha_j - self.alpha[j]) * self.X[i] @ self.X[j].T

        new_b2 = self.b - self.E[j] - self.y[i] * (new_alpha_i - self.alpha[i]) * self.X[i] @ self.X[j].T \
                 - self.y[j] * (new_alpha_j - self.alpha[j]) * self.X[j] @ self.X[j].T

        if 0 < new_alpha_i < self.C:
            new_b = new_b1
        elif 0 < new_alpha_j < self.C:
            new_b = new_b2
        else:
            new_b = (new_b1 + new_b2) / 2.0

        # Update alpha and error cache
        self.alpha[i] = new_alpha_i
        self.alpha[j] = new_alpha_j

        self.update_error_cache(i)
        self.update_error_cache(j)

        self.b = new_b

        return True

    def compute_L_H(self, i, j):
        # Compute bounds L and H
        if self.y[i] != self.y[j]:
            L = max(0, self.alpha[j] - self.alpha[i])
            H = min(self.C, self.C + self.alpha[j] - self.alpha[i])
        else:
            L = max(0, self.alpha[i] + self.alpha[j] - self.C)
            H = min(self.C, self.alpha[i] + self.alpha[j])

        return L, H

    def clip_alpha(self, alpha, L, H):
        # Clip alpha to be within the range [L, H]
        if alpha < L:
            return L
        elif alpha > H:
            return H
        else:
            return alpha

    def calculate_b(self):
        # Compute the bias term
        support_vectors_indices = np.where((self.alpha > 0) & (self.alpha < self.C))[0]

        if len(support_vectors_indices) == 0:
            return 0.0  # Handle the case when there are no support vectors

        b_sum = 0.0
        for i in support_vectors_indices:
            b_sum += self.y[i] - self.decision_function(self.X[i])

        return b_sum / len(support_vectors_indices)

    def predict(self, X):
        decision_values = self.decision_function(X)
        predictions = np.where(decision_values > 0, 1, 0)
        return predictions.astype(int) 