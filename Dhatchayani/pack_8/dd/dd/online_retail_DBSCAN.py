import numpy as np

class CustomDBSCAN:
    def __init__(self, eps, min_samples):
        self.eps = eps
        self.min_samples = min_samples
        self.labels_ = None

    def fit(self, X):
        self.labels_ = np.zeros(len(X))

        # Initialize a cluster label
        current_label = 0

        # Iterate through each data point
        for i in range(len(X)):
            if self.labels_[i] != 0:
                continue  # Skip already processed points

            # Find neighbors
            neighbors = self._find_neighbors(X, i)

            if len(neighbors) < self.min_samples:
                # Assign noise label (-1)
                self.labels_[i] = -1
            else:
                current_label += 1
                # Expand cluster
                self._expand_cluster(X, i, neighbors, current_label)

    def _find_neighbors(self, X, index):
      distances = np.linalg.norm(X.values - X.values[index], axis=1)
      return [i for i, d in enumerate(distances) if 0 < d < self.eps]


    def _expand_cluster(self, X, index, neighbors, current_label):
        # Expand the cluster starting from a core point
        self.labels_[index] = current_label

        i = 0
        while i < len(neighbors):
            neighbor = neighbors[i]
            if self.labels_[neighbor] == -1:
                self.labels_[neighbor] = current_label
            elif self.labels_[neighbor] == 0:
                self.labels_[neighbor] = current_label
                new_neighbors = self._find_neighbors(X, neighbor)
                if len(new_neighbors) >= self.min_samples:
                    neighbors += new_neighbors
            i += 1

custom_dbscan = CustomDBSCAN(eps=0.5, min_samples=4)
custom_dbscan.fit(RFM[['Recency', 'Frequency', 'Monetary', 'Clusters']])
