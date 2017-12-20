import model_building
import numpy as np

# local libraries
import route.config as conf

class ModelBuilder(object):
    def __init__(self, *args, **kwargs):
        super(ModelBuilder, self).__init__()
        self.conf_embedding = conf.load("embedding")
        self.kwargs = kwargs
        try:
            str(kwargs["method"])
        except:
            print "You must specify a topic modeling method! Only tfidf and lda is supported now."
        else:
            self.method = str(kwargs["method"])
            if self.method not in ['tfidf', 'lda']:
                print "Error. Only tfidf and lda is supported now. We will use method=tfidf in the following analysis."
                self.method = 'tfidf'
            self.CONF_EMBEDDING_DETAILS = self.conf_embedding[self.method]
        pass

    def buildModelAndSave(self, *args, **kwargs):
        if self.kwargs["method"] == "tfidf":
            self.tfidf = model_building.TfidfEmbedding.TfidfEmbedding(
                token_source = self.conf_embedding["prefix"] + self.CONF_EMBEDDING_DETAILS["token_source"],
                num_to_train = self.CONF_EMBEDDING_DETAILS["num_to_train"],
                dict_save_to = self.conf_embedding["prefix"] + self.CONF_EMBEDDING_DETAILS["dict_save_to"],
                corpus_save_to = self.conf_embedding["prefix"] + self.CONF_EMBEDDING_DETAILS["corpus_save_to"],
                model_save_to = self.conf_embedding["prefix"] + self.CONF_EMBEDDING_DETAILS["model_save_to"]
            )
            self.dictionary = self.tfidf.dictionary
            self.token_list = self.tfidf.token_list
            return self.tfidf
        elif self.kwargs["method"] == "lda":
            num_topics = int(kwargs["num_topics"])
            chunksize = int(kwargs["chunksize"])
            self.lda = model_building.LdaEmbedding.LdaEmbedding(
                token_source = self.conf_embedding["prefix"] + self.CONF_EMBEDDING_DETAILS["token_source"],
                num_to_train = self.CONF_EMBEDDING_DETAILS["num_to_train"],
                dict_save_to = self.conf_embedding["prefix"] + self.CONF_EMBEDDING_DETAILS["dict_save_to"],
                corpus_save_to = self.conf_embedding["prefix"] + self.CONF_EMBEDDING_DETAILS["corpus_save_to"],
                model_save_to = self.conf_embedding["prefix"] + self.CONF_EMBEDDING_DETAILS["model_save_to"],
                num_topics = num_topics,
                chunksize = chunksize
            )
            self.dictionary = self.lda.dictionary
            self.token_list = self.lda.token_list
            return self.lda

    def buildSimilarityMatrix(self):
        if self.kwargs["method"] == "tfidf":
            self.tfidf_sim_mat = self.tfidf._buildSimilarityMat()
            self.corpus_tfidf = self.tfidf.corpus_tfidf
            self.tfidf._saveCorpus(self.conf_embedding["prefix"] + self.CONF_EMBEDDING_DETAILS["sim_index_save_to"])
            return self.tfidf_sim_mat
        elif self.kwargs["method"] == "lda":
            self.lda_sim_mat = self.lda._buildSimilarityMat()
            self.corpus_lda = self.lda.corpus_lda
            self.lda._saveCorpus(self.conf_embedding["prefix"] + self.CONF_EMBEDDING_DETAILS["sim_index_save_to"])
            return self.lda_sim_mat

    def getEmbeddingMatrix(self, *args, **kwargs):
        if self.kwargs["method"] == "tfidf":
            if kwargs["build"] == True:
                self.buildSimilarityMatrix()
                self.embedding_matrix = np.zeros([self.tfidf.tfidf.num_docs, len(self.dictionary.values())])
                for ind, transformed_corpus in enumerate(self.corpus_tfidf):
                    for token in transformed_corpus:
                        self.embedding_matrix[ind][token[0]] = token[1]
            elif kwargs["build"] == False:
                self.embedding_matrix = np.zeros([self.tfidf.tfidf.num_docs, len(self.dictionary.values())])
                for ind, transformed_corpus in enumerate(self.corpus_tfidf):
                    for token in transformed_corpus:
                        self.embedding_matrix[ind][token[0]] = token[1]
            return self.embedding_matrix
        elif self.kwargs["method"] == "lda":
            if kwargs["build"] == True:
                self.buildSimilarityMatrix()
                self.embedding_matrix = np.zeros([self.lda.lda.num_docs, len(self.dictionary.values())])
                for ind, transformed_corpus in enumerate(self.corpus_lda):
                    for token in transformed_corpus:
                        self.embedding_matrix[ind][token[0]] = token[1]
            elif kwargs["build"] == False:
                self.embedding_matrix = np.zeros([self.lda.lda.num_docs, len(self.dictionary.values())])
                for ind, transformed_corpus in enumerate(self.corpus_lda):
                    for token in transformed_corpus:
                        self.embedding_matrix[ind][token[0]] = token[1]
            return self.embedding_matrix





