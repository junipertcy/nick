import os
import os.path
import json

from gensim import corpora, models
import pynlpir
from collections import defaultdict

from nltk.tokenize import StanfordSegmenter
from nltk.tokenize import StanfordTokenizer

# local libraries
from learning.tag import StanfordPOSTagger
import preprocessing as pre
import route.config as conf
from utils import Timer

import logging
logging.basicConfig(
    ormat='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO
)


class KeywordExtractor(object):
    def __init__(self, **kwargs):
        self.conf_io = conf.load("io")
        self.conf_corenlp = conf.load("stanford_corenlp")
        self.conf_embedding = conf.load("embedding")
        conf_segmenter = self.conf_corenlp["segmenter"]
        conf_tokenizer = self.conf_corenlp["tokenizer"]
        conf_postagger = self.conf_corenlp["postagger"]
        prefix = self.conf_corenlp["prefix"]

        self.segmenter = StanfordSegmenter(
            path_to_jar=prefix + conf_segmenter["path_to_jar"],
            path_to_sihan_corpora_dict=prefix + conf_segmenter["path_to_sihan_corpora_dict"],
            path_to_model=prefix + conf_segmenter["path_to_model"],
            path_to_dict=prefix + conf_segmenter["path_to_dict"],
            path_to_slf4j=prefix + conf_segmenter["path_to_slf4j"],
            encoding=conf_segmenter["encoding"]
        )
        self.enTokenizer = StanfordTokenizer(
            path_to_jar=prefix + conf_tokenizer["path_to_jar"]
        )
        self.zh_tagger = StanfordPOSTagger(
            prefix + conf_postagger["tagger_zh"],
            path_to_jar=prefix + conf_postagger["path_to_jar"]
        )
        self.en_tagger = StanfordPOSTagger(
            prefix + conf_postagger["tagger_en"],
            path_to_jar=prefix + conf_postagger["path_to_jar"]
        )
        self.frequency = defaultdict(int)
        pynlpir.open()
        pynlpir.nlpir.ImportUserDict(conf.load("pynlpir")["user_dict"], Overwrite = False)

        try:
            self.excluded_docs = kwargs["excluded_docs"]
        except:
            self.excluded_docs = [""]

        # experimental features
        self.f_token_indexes = prefix + conf.load("pynlpir")["user_dict"]

    def _getDocs(self, num_docs, **kwargs):
        # from pymongo import MongoClient
        # dbClient = MongoClient("mongodb://127.0.0.1:27017")
        # db = dbClient["nick"]
        # collection = db["tickets"]
        # cursor = collection.find({}).limit(num_docs)
        # return enumerate(cursor)
        separated = kwargs["separated"]
        i = 0
        if separated is True:
            samples_dir = conf.load('samples')['dir']
            docs = [os.path.join(samples_dir, x) for x in os.listdir(samples_dir)]
            while i < num_docs:
                with open(docs[i]) as f:
                    try:
                        yield i, json.load(f)
                        i += 1
                    except:
                        i += 1  # TODO: may cause problem on the generator index
                        num_docs += 1
            return
        else:
            samples_loc = conf.load('samples')['single']
            docs = json.loads(open(samples_loc).read())
            while i < num_docs:
                try:
                    yield i, docs[i]
                    i += 1
                except:
                    i += 1  # TODO: may cause problem on the generator index
                    num_docs += 1
            return
        #try:
        #    from pymongo import MongoClient
        #    dbClient = MongoClient("mongodb://127.0.0.1:27017")
        #    db = dbClient["nick"]
        #    collection = db["tickets"]
        #    cursor = collection.find({}).limit(num_docs)
        #    return enumerate(cursor)
        #except ImportError:
        #    i = 0
        #    docs = [os.path.join(conf.load("samples"), x) for x in os.listdir(conf.load("samples"))]
        #    while i < num_docs:
        #        with open(docs[i]) as f:
        #            i += 1
        #            yield json.load(f)
        #    return

    def saveToDoclist(self, num_docs, **kwargs):
        file_docs = open(self.conf_io["prefix"] + self.conf_io["output_data_directory"] + 'num_docs-' + str(num_docs) + '.doclist', 'wb')
        separated = kwargs["separated"]
        docs = self._getDocs(num_docs, separated=separated)
        for ind, i in docs:
            try:
                text = i["title"].replace("\n", " ").replace("\r", " ") + " " + i["body"].replace('\n', ' ').replace("\r", " ")
            except Exception as e:
                print e
                continue
            else:
                file_docs.write("%s\n" % (text.encode("utf-8")))
        file_docs.close()

    def calculateTokens(self, line, **kwargs):
        line = """{}""".format(line)
        doc = [excluded_doc for excluded_doc in self.excluded_docs if excluded_doc not in line.split('\n')[0]]
        if doc == []:
            print "skipped:", line.split('\n')[0]
            return ''

        doc_len_lower_bound = int(kwargs["doc_len_lower_bound"])
        doc_len_upper_bound = int(kwargs["doc_len_upper_bound"])
        if len(line.decode('utf-8')) <= doc_len_lower_bound or len(line.decode('utf-8')) >= doc_len_upper_bound:
            return ''

        allowed_list = ["noun", "intransitive verb", "noun-verb", "adjective"]
        # if you want to try the stanford coreNLP tokenizer in other languages...
        _en_tokens = [token.lower() for token in pre.GetEnTokens()(line)]
        with Timer('stanford_seg') as t:
            _en_tokens_tokenized = self.enTokenizer.tokenize(' '.join(_en_tokens))
        en_tokens = [token for token in _en_tokens_tokenized if token.lower() not in pre.FilterKeywords().getStopwordsAsJSON()["en"]]
        en_tokens = [token for token in en_tokens if token.lower() not in pre.FilterKeywords().getCustomStopwordsAsList()]
        en_tokens = list(set(en_tokens))
        # now we have English tokens...
        tokens_in_each_doc = []
        with Timer('stanford_tag') as t:
            tags = self.en_tagger.tag(en_tokens)
        for word, tag in tags:
            if tag in ["NN", "FW", "VBD", "NNS", "VBP"]:
                tokens_in_each_doc.append(word)

        # _token_list = [i[0] for i in pynlpir.get_key_words(line.decode("utf-8"), weighted=True)] + en_tokens
        if str(kwargs["method"]) == "keyword":
            _token_list = [i[0] for i in pynlpir.get_key_words(line.decode("utf-8"), weighted=True)]
        elif str(kwargs["method"]) == "normal":
            # for i in pynlpir.segment(line.decode("utf-8"), pos_names='child'):
                # print i[0], i[1]
            if "2G" in line.decode("utf-8"):        # hot fix for a bug
                line = line.replace("2G", "")
                _token_list = [i[0] for i in pynlpir.segment(line.decode("utf-8"), pos_names='child') if i[1] in allowed_list]
            else:
                _token_list = [i[0] for i in pynlpir.segment(line.decode("utf-8"), pos_names='child') if i[1] in allowed_list]
        __token_list = [token for token in _token_list if token not in pre.FilterKeywords().getStopwordsAsJSON()["zh"]]

        token_list = [token for token in __token_list
            if token.lower() not in pre.FilterKeywords().getStopwordsAsJSON()["en"]
            and token.lower() not in pre.FilterKeywords().getCustomStopwordsAsList()]
        zh_tokens = [token for token in token_list if token not in _en_tokens]
        token_list = zh_tokens + tokens_in_each_doc

        #remove item in token_list that appears only few times
        for token in token_list:
            self.frequency[token.lower()] += 1
        tokens = ','.join(token_list)
        print "Done tokenizing text: ", tokens

        return tokens

    def getKeywordsAndSave(self, *args, **kwargs):
        import pickle
        freq_lower_bound = int(kwargs["freq_lower_bound"])
        token_len_lower_bound = int(kwargs["token_len_lower_bound"])
        doc_len_lower_bound = int(kwargs["doc_len_lower_bound"])
        doc_len_upper_bound = int(kwargs["doc_len_upper_bound"])

        if str(kwargs["method"]) == "keyword":
            file_keywords = open(self.conf_io["prefix"] + self.conf_io["output_data_directory"] + str(kwargs["target_name"]) + '.fine.keywords', 'w')
        elif str(kwargs["method"]) == "normal":
            file_keywords = open(self.conf_io["prefix"] + self.conf_io["output_data_directory"] + str(kwargs["target_name"]) + '.keywords', 'w')
        tokens = []
        token_indexes = {}
        if bool(kwargs["static_file"]) is True:
            source_name = self.conf_io["prefix"] + self.conf_io["output_data_directory"] + str(kwargs["source_name"])
            with open(source_name, 'r') as f:
                _ind = 0
                for ind, line in enumerate(f):
                    try:
                        with Timer('calculateTokens') as t:
                            tokens.append(self.calculateTokens(
                                line,
                                method=str(kwargs["method"]),
                                doc_len_lower_bound=doc_len_lower_bound,
                                doc_len_upper_bound=doc_len_upper_bound
                            ))
                        # [experimental feature]
                        # this is to be used with LDA
                        # to show what raw doc is associated with each topic
                        token_indexes[ind] = _ind
                        _ind += 1
                    except Exception as e:
                        if e is KeyboardInterrupt:
                            break
                        print e
                        print "error with ", line
                        continue
                    else:
                        pass
                for line in tokens:
                    if line is not None:
                        filtered_tokens = [token for token in line.split(',') if self.frequency[token.lower()] > freq_lower_bound and len(token) > token_len_lower_bound]
                        filtered_tokens = ','.join(filtered_tokens)
                        file_keywords.write('%s\n' % (filtered_tokens.encode('utf-8')))
                        file_keywords.flush()
            f.close()
            # experimental
            json.dump(token_indexes, open(self.f_token_indexes + "token_indexes.pickle", "w"), ensure_ascii=True)
        else:
            doc_list = args[0]
            for ind, line in enumerate(list(doc_list)):
                try:
                    tokens.append(self.calculateTokens(
                        line,
                        method=str(kwargs["method"]),
                        doc_len_lower_bound=doc_len_lower_bound,
                        doc_len_upper_bound=doc_len_upper_bound
                    ))
                except Exception as e:
                    if e is KeyboardInterrupt:
                        break
                    print e
                    print "error with ", line
                    continue
                else:
                    pass
            for line in tokens:
                if line is not None:
                    filtered_tokens = [token for token in line.split(',') if self.frequency[token.lower()] > freq_lower_bound and len(token) > token_len_lower_bound]
                    filtered_tokens = ','.join(filtered_tokens)
                    file_keywords.write('%s\n' % (filtered_tokens.encode('utf-8')))
                    file_keywords.flush()
        file_keywords.close()
        pynlpir.close()
        return True

    def _loadTopicModel(self, **kwargs):
        try:
            str(kwargs["method"])
        except:
            print "You must specify a topic modeling method! Only tfidf is supported now."
        else:
            self.method = str(kwargs["method"])
            if self.method != 'tfidf':
                print "Error. We will use method=tfidf in the following analysis."
                self.method = 'tfidf'
            self.conf_tfidf = self.conf_embedding[self.method]

        _corpora = corpora.MmCorpus(self.conf_embedding["prefix"] + self.conf_tfidf["corpus_save_to"] + '.mm')
        self.dictionary = corpora.Dictionary.load(self.conf_embedding["prefix"] + self.conf_tfidf["dict_save_to"] + '.dict')

        _model = models.TfidfModel.load(self.conf_embedding["prefix"] + self.conf_tfidf["model_save_to"] + '.tfidf.model')

        return _model, _corpora

    def refineKeywords(self, **kwargs):
        #TODO: Whether setting TF-IDF threshold?
        top_k = int(kwargs["top_k"])
        file_keywords = open(self.conf_io["prefix"] + self.conf_io["output_data_directory"] + str(kwargs["target_name"]) + '.filtered.keywords', 'w')
        _model, _corpora = self._loadTopicModel(method = 'tfidf')

        for corpus in _corpora:
            # take the top-10 tf-idf weight tokens within each document, also, we set an absolute weight to it
            corpus = _model[corpus]
            sorted_corpus_per_doc = [token for token in sorted(corpus, key=lambda x: -x[1])[:top_k]]
            tokens = [self.dictionary.id2token[_token[0]] for _token in sorted_corpus_per_doc]
            tokens = ','.join(tokens)
            file_keywords.write('%s\n' % (tokens.encode('utf-8')))

        file_keywords.close()
        return True
