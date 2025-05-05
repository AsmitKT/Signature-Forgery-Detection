import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def cluster_and_plot(embeddings: np.ndarray,
                     eps: float = 0.5,
                     min_samples: int = 5):
    """
    embeddings: (N, D) array of feature vectors
    returns: array of cluster labels of length N
    Also shows a 2D scatter via PCA + coloring by cluster.
    """
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(embeddings)
    labels     = clustering.labels_

    # reduce to 2D for visualization
    pca     = PCA(n_components=2)
    reduced = pca.fit_transform(embeddings)

    plt.figure(figsize=(6,6))
    scatter = plt.scatter(reduced[:,0], reduced[:,1],
                          c=labels, cmap="tab10", s=30)
    plt.title("DBSCAN Clustering of Signature Embeddings")
    plt.xlabel("PC 1")
    plt.ylabel("PC 2")
    plt.colorbar(scatter, label="Cluster Label")
    plt.show()

    return labels
