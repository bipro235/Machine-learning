
from sklearn.cluster import SpectralClustering
import numpy as np

# Load data
X = np.array([[1, 2], [1, 4], [1, 0],
              [4, 2], [4, 4], [4, 0]])

# Spectral clustering
clustering = SpectralClustering(n_clusters=2,
                                assign_labels='discretize',
                                random_state=0).fit(X)

print('Cluster labels:', clustering.labels_)