
import numpy as np

def calculate_distance(point1, point2):
    return np.sqrt(np.sum((point1 - point2) ** 2))

def kmeans(X, k, max_iters=100):
    # Initialize centroids randomly
    centroids = X[np.random.choice(X.shape[0], k, replace=False)]

    for _ in range(max_iters):
        # Assign each example to the nearest centroid
        labels = np.argmin(np.array([[calculate_distance(x, centroid) for centroid in centroids] for x in X]), axis=1)

        # Update centroids based on mean of assigned examples
        new_centroids = np.array([X[labels == j].mean(axis=0) for j in range(k)])

        # Check for convergence
        if np.array_equal(new_centroids, centroids):
            break

        centroids = new_centroids

    return labels, centroids
