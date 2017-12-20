import clustering as cl
import warnings
from sklearn.cluster import MiniBatchKMeans


class DocumentClusterer(object):

    def __init__(self, *args, **kwargs):
        super(DocumentClusterer, self).__init__()
        pass

    def determineK(self, embedding_matrix, **kwargs):
        best_K = cl.DetK()(
            embedding_matrix,
            range_n_clusters_min=kwargs["range_n_clusters_min"],
            range_n_clusters_max=kwargs["range_n_clusters_max"]
        )
        return best_K

    def fitClusters(self, embedding_matrix, **kwargs):
        self.km = MiniBatchKMeans(n_clusters=kwargs["n_clusters"])

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=DeprecationWarning)
            self.km.fit(embedding_matrix)
        return self.km

    def getClusters(self):
        return self.km.labels_.tolist()
