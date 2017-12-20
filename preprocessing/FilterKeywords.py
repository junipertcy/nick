import json
import route.config as conf

class FilterKeywords(object):
    def __init__(self):
        self.conf_stopwords = conf.load("stopwords")
        self.prefix = self.conf_stopwords["prefix"]
        pass

    def getStopwordsAsJSON(self):
        with open(self.prefix + self.conf_stopwords["general"], 'rb') as data_file:
            stopwords = json.load(data_file)
        data_file.close()
        return stopwords

    def getCustomStopwordsAsList(self):
        stopwords = []
        with open(self.prefix + self.conf_stopwords["custom"], 'rb') as data_file:
            for word in data_file:
                stopwords.append(word.split('\n')[0])
        data_file.close()
        return stopwords