import sys
from collections import Counter

from nltk.tokenize import StanfordTokenizer

# local libraries
from learning.tag import StanfordPOSTagger
import route.config as conf

import pynlpir
import regex

# tfidf
from sklearn.feature_extraction.text import TfidfVectorizer


class WordSegment(object):
    def __init__(self, user_dict=None):
        self.conf_io = conf.load("io")
        self.conf_corenlp = conf.load("stanford_corenlp")
        self.conf_embedding = conf.load("embedding")
        conf_tokenizer = self.conf_corenlp["tokenizer"]
        conf_postagger = self.conf_corenlp["postagger"]
        prefix = self.conf_corenlp["prefix"]

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

        # TODO:
        # 這裡要加上自定義字典

    def get_tokens(self, text):
        tokens = self.enTokenizer.tokenize(text)

        return self.en_tagger.tag(tokens)

    def get_new_words(self, text):
        pass


class ZhWordSegment(object):
    def __init__(self, user_defined_words=None):
        if user_defined_words:
            for word in user_defined_words:
                pynlpir.nlpir.AddUserWord(word)
        pynlpir.open()

    def get_tokens(self, text):
        return pynlpir.segment(text, pos_names='child')

    def get_new_words(self, text):
        pass


class Tokenizer:
    ALLOW_POS_ZH = ["noun", "intransitive verb", "noun-verb", "adjective"]
    ALLOW_POS_NON_ZH = ["NN", "FW", "VBD", "NNS", "VBP"]

    def __init__(self, zh_dict=None, non_zh_dict=None):
        self.zh_word_segment = ZhWordSegment(zh_dict)
        self.non_zh_word_segment = WordSegment(non_zh_dict)

    def __call__(self, text):
        # non_zh_text = regex.sub(r'[\p{Han}\p{P}]+', ' ', text)
        # non_zh_tokens = self.non_zh_word_segment.get_tokens(non_zh_text)
        zh_text = regex.sub(r'[^\p{Han}\p{P}]+', ' ', text)
        zh_tokens = self.zh_word_segment.get_tokens(zh_text)
        return list(filter(lambda x: x[1] in self.ALLOW_POS_ZH, zh_tokens))

        # return list(filter(lambda x: x[1] in self.ALLOW_POS_ZH, zh_tokens)) + \
        #        list(filter(lambda x: x[1] in self.ALLOW_POS_NON_ZH, non_zh_tokens))


class KeyTermExtractor:
    def __init__(self, model=None, zh_dict=None, non_zh_dict=None):
        self.tokenizer = Tokenizer(zh_dict, non_zh_dict)
        self.vectorizer = model or TfidfVectorizer(min_df=2, analyzer='word', tokenizer=self.tokenizer)

    def get_model(self):
        return self.vectorizer

    def fit(self, docs):
        self.vectorizer.fit(docs)

    def extract(self, doc, top=5):
        words = self.vectorizer.get_feature_names()
        tfidf = self.vectorizer.transform([doc])
        words_with_weight = []
        for i in range(len(words)):
            if tfidf[0, i] > 0:
                words_with_weight.append((words[i], tfidf[0, i]))
        return sorted(words_with_weight, reverse=True, key=lambda x: x[1])[:top]


if __name__ == '__main__':
    docs = []
    with open(sys.argv[1], 'r') as f:
        for line in f:
            docs.append(line)

    extractor = KeyTermExtractor()
    extractor.fit(docs)

    tokens = []
    for text in docs:
        tokens += list(map(lambda x: x[0], extractor.extract(text)))
    print(Counter(tokens).most_common(5))
