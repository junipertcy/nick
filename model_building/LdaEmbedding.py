from gensim import corpora, models, similarities


class LdaEmbedding(object):

    def _getTokenList(self, filename, number):
        '''
        filename: filename of keywords
        number: number of corpus to train
        '''
        token_list = []
        with open(filename, "r") as f:
            counter = 0
            for ind, line in enumerate(f):
                text = line
                text = text.split('\n')[0]
                try:
                    text.decode("utf-8")
                except UnicodeDecodeError:
                    print 'UnicodeDecodeError', line
                    continue
                else:
                    token_list += [text.decode("utf-8").lower().split(',')]
                counter += 1
                if counter == number:
                    break
        f.close()
        token_list = [token for token in token_list if token != [""]]
        return token_list

    def _buildDictionary(self):
        dictionary = corpora.Dictionary(self.token_list)
        return dictionary

    def _saveDictionary(self, target):
        try:
            self.dictionary.save(target)
        except Exception as e:
            print "save error: ", e
            return False
        else:
            return True

    def _buildCorpus(self):
        try:
            self.corpus = [
                self.dictionary.doc2bow(text) for text in self.token_list
            ]
        except Exception as e:
            print "build corpus error: ", e
            return False
        else:
            return self.corpus

    def _saveCorpus(self, target):
        try:
            corpora.MmCorpus.serialize(target, self.corpus)
        except Exception as e:
            print "save corpus error: ", e
            return False
        else:
            return True

    def transformCorpus(self):
        try:
            self.corpus_lda = self.lda[self.corpus]
        except Exception as e:
            print "transform corpus with error: ", e
            return False
        else:
            print "We convert our vectors corpus to LDA space : ", type(self.corpus_lda)
            return self.corpus_lda

    def __init__(self, *args, **kwargs):
        token_source = str(kwargs["token_source"])
        num_to_train = int(kwargs["num_to_train"])
        dict_save_to = str(kwargs["dict_save_to"])
        corpus_save_to = str(kwargs["corpus_save_to"])
        model_save_to = str(kwargs["model_save_to"])
        num_topics = int(kwargs["num_topics"])
        chunksize = int(kwargs["chunksize"])

        self.token_list = self._getTokenList(token_source, num_to_train)
        self.dictionary = self._buildDictionary()
        self.dictionary.items()  # initiate id2token mapping
        self.corpus = self._buildCorpus()

        self._saveDictionary(dict_save_to + '.dict')
        self._saveCorpus(corpus_save_to + '.mm')

        self.lda = models.ldamodel.LdaModel(
            corpus=self.corpus,
            id2word=self.dictionary,
            update_every=1,
            chunksize=chunksize,
            passes=1,
            alpha='auto',
            num_topics=num_topics
        )

        self.lda.save(model_save_to + '.lda.model')
        print "Awesome! The LDA vector space is constructed!!"
        print self.lda

    def _buildSimilarityMat(self):
        try:
            self.corpus_lda = self.lda[self.corpus]
            self.sim_index = similarities.MatrixSimilarity(self.corpus_lda)
        except Exception as e:
            print "generate similarity matrice error: ", e
            return False
        else:
            print "We built the LDA similarities index from the corpus :", type(self.sim_index)
            self.sim_mat = self.sim_index[self.corpus_lda]
            print "We get a similarity matrix for all documents in the corpus: ", type(self.sim_mat)
            return self.sim_mat

    def _saveSimilarityMat(self, target):
        try:
            self.sim_index.save(target + '.index')
        except Exception as e:
            print "save similarity index error: ", e
            return False
        else:
            print "LDA similarity matrix saved."
            return True
