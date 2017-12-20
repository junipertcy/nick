from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score

import numpy as np
import warnings

class DetK(object):

    def __init__(self):
        pass

    def __call__(self, X, range_n_clusters_min, range_n_clusters_max):
        range_n_clusters = range(range_n_clusters_min, range_n_clusters_max)

        silhouette_avg_list = []
        for ind, n_clusters in enumerate(range_n_clusters):

            clusterer = MiniBatchKMeans(n_clusters=n_clusters, random_state=10)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=DeprecationWarning)
                cluster_labels = clusterer.fit_predict(X)

            silhouette_avg = silhouette_score(X, cluster_labels)
            print("For n_clusters =", n_clusters,
                  "The average silhouette_score is :", silhouette_avg)
            silhouette_avg_list.append((range_n_clusters[ind], silhouette_avg))

        best_K = sorted(silhouette_avg_list, key = lambda x: -x[1])[0][0]
        return best_K
