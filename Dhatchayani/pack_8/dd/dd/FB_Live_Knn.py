import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score

class KMeansClustering:
    def __init__(self, n_clusters, n_iterations=100, random_state=None):
        self.n_clusters = n_clusters
        self.n_iterations = n_iterations
        self.random_state = random_state
        self.centroids = None
        self.labels = None

    def initialize_centroids(self, X):
        if self.random_state is not None:
            np.random.seed(self.random_state)
        return X.sample(self.n_clusters, random_state=self.random_state).values

    def assign_labels(self, X, centroids):
        return np.argmin(np.linalg.norm(X.values[:, np.newaxis] - centroids, axis=2), axis=1)

    def update_centroids(self, X, labels):
        return np.array([X.values[labels == i].mean(axis=0) for i in range(self.n_clusters)])

    def fit(self, X):
        centroids = self.initialize_centroids(X)

        for _ in range(self.n_iterations):
            labels = self.assign_labels(X, centroids)
            new_centroids = self.update_centroids(X, labels)

            if np.allclose(centroids, new_centroids, atol=1e-4):
                break

            centroids = new_centroids

        self.labels = labels
        self.centroids = centroids

    def silhouette_score(self, X):
        return silhouette_score(X, self.labels)

# Assuming 'X' is your feature matrix
# Convert 'y' to a Pandas Series
y = pd.Series(y)
y.reset_index(drop=True, inplace=True)

# Initialize KMeansClustering object
kmeans_clustering = KMeansClustering(n_clusters=4, random_state=0)

# Fit the model
kmeans_clustering.fit(X)

# Get the labels and centroids
labels = kmeans_clustering.labels
centroids = kmeans_clustering.centroids

# Calculate silhouette score
silhouette_avg = kmeans_clustering.silhouette_score(X)
print("Silhouette Score:", silhouette_avg)
