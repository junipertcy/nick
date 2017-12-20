from gensim import corpora, models, similarities


class TfidfEmbedding(object):

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
            self.corpus = [self.dictionary.doc2bow(text) for text in self.token_list]
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
            self.corpus_tfidf = self.tfidf[self.corpus]
        except Exception as e:
            print "transform corpus with error: ", e
            return False
        else:
            print "We convert our vectors corpus to TF-IDF space : ", type(corpus_tfidf)
            return self.corpus_tfidf

    def __init__(self, *args, **kwargs):
        token_source = str(kwargs["token_source"])
        num_to_train = int(kwargs["num_to_train"])
        dict_save_to = str(kwargs["dict_save_to"])
        corpus_save_to = str(kwargs["corpus_save_to"])
        model_save_to = str(kwargs["model_save_to"])

        self.token_list = self._getTokenList(token_source, num_to_train)
        self.dictionary = self._buildDictionary()
        self.dictionary.items() #initiate id2token mapping
        self.corpus = self._buildCorpus()

        self._saveDictionary(dict_save_to + '.dict')
        self._saveCorpus(corpus_save_to + '.mm')

        self.tfidf = models.TfidfModel(self.corpus)
        self.tfidf.save(model_save_to + '.tfidf.model')
        print "Awesome! The TF-IDF vector space is constructed!!"
        print self.tfidf

    def _buildSimilarityMat(self):
        try:
            self.corpus_tfidf = self.tfidf[self.corpus]
            self.sim_index = similarities.MatrixSimilarity(self.corpus_tfidf)
        except Exception as e:
            print "generate similarity matrice error: ", e
            return False
        else:
            print "We built the TF-IDF similarities index from the corpus :", type(self.sim_index)
            self.sim_mat = self.sim_index[self.corpus_tfidf]
            print "We get a similarity matrix for all documents in the corpus: ", type(self.sim_mat)
            return self.sim_mat

    def _saveSimilarityMat(self, target):
        try:
            self.sim_index.save(target + '.index')
        except Exception as e:
            print "save similarity index error: ", e
            return False
        else:
            print "TF-IDF similarity matrix saved."
            return True
