from nick import KeywordExtractor as keyExt
from tgrocery import Grocery

from sklearn.metrics import recall_score as recall_score
from sklearn.utils import shuffle
import os
import route.config as conf


class TagPredictor(object):
    def _custom_tokenize(self, line, **kwargs):
        try:
            kwargs["method"]
        except:
            method = str(self.kwargs["method"])
        else:
            method = str(kwargs["method"])
        if method == "normal":
            tokens = self.key_ext.calculateTokens(
                line,
                doc_len_lower_bound=5,
                doc_len_upper_bound=500,
                method="normal"
            )
        elif method == "processed":
            tokens = line.split(',')
        return tokens

    def __init__(self, *args, **kwargs):
        self.grocery_name = str(kwargs["grocery_name"])
        method = str(kwargs["method"])
        train_src = str(kwargs["train_src"])
        self.PREFIX = conf.load("predict_label")["prefix"]
        self.MODEL_DIR = conf.load("predict_label")["model_dir"]

        self.kwargs = kwargs
        if method == "normal":
            self.key_ext = keyExt()
            self.grocery = Grocery(self.grocery_name, custom_tokenize = self._custom_tokenize)
        elif method == "jieba":
            self.grocery = Grocery(self.grocery_name)
        elif method == "processed":
            self.grocery = Grocery(self.grocery_name, custom_tokenize = self._custom_tokenize)
        pass

    def trainFromDocs(self, *args, **kwargs):
        model = self.grocery.train(self.kwargs["train_src"])
        return model

    def autoEvaluation(self, *args, **kwargs):
        prune_threshold = float(kwargs["threshold"])
        excluded_labels = kwargs["excluded_labels"]
        excluded_docs = kwargs["excluded_docs"]

        train_data = []
        with open(self.kwargs["train_src"], 'rb') as f:
            for line in f:
                try:
                    line.split('\t', 1)[1]
                except:
                    continue
                else:
                    train_data.append((line.split('\t', 1)[0], line.split('\t', 1)[1].split('\n', 1)[0]))
        f.close()

        print "#items before filtering:", len(train_data)
        print "-- Now we filter out the excluded docs --"
        train_data = [i for i in train_data if i[1] not in excluded_docs]
        print "#items after filtering:", len(train_data)
        print "-- Now we filter out the excluded labels --"
        train_data = [i for i in train_data if i[0] not in excluded_labels]
        print "#items after filtering:", len(train_data)

        n = len(train_data) #number of rows in your dataset
        indices = range(n)
        indices = shuffle(indices)
        train_set = map(lambda x: train_data[x], indices[:n*10//10])
        test_set = map(lambda x: train_data[x], indices[:n*10//10])

        self.grocery.train(train_set)
        test_result = self.grocery.test(test_set)
        print '-- Accuracy after training --'
        print 'Accuracy, A-0:', test_result

        low_recall_label = []
        for item in test_result.recall_labels.items():
            if item[1] < prune_threshold:
                low_recall_label.append(item[0])
        new_train_set = [item for item in train_set if item[0] not in low_recall_label]
        new_test_set = [item for item in train_set if item[0] not in low_recall_label]

        self.grocery.train(new_train_set)
        new_test_result = self.grocery.test(new_test_set)

        print '-- Accuracy after training, with low-recall labels (less than', str(prune_threshold*100) , '%) pruned --'
        print 'Accuracy, A-1:', new_test_result

        return self.grocery, new_test_result

    def manualEvaluation(self, *args, **kwargs):
        n_docs = int(kwargs["n_docs"])
        excluded_labels = kwargs["excluded_labels"]
        excluded_docs = kwargs["excluded_docs"]

        train_data = []
        with open(self.kwargs["train_src"], 'rb') as f:
            for line in f:
                try:
                    line.split('\t', 1)[1]
                except:
                    continue
                else:
                    train_data.append((line.split('\t', 1)[0], line.split('\t', 1)[1].split('\n', 1)[0]))
        f.close()

        train_data = [item for item in train_data if item[0] not in excluded_labels]
        train_data = [i for i in train_data if i[1] not in excluded_docs]

        n = len(train_data) #number of rows in your dataset
        indices = range(n)
        indices = shuffle(indices)
        test_set = map(lambda x: train_data[x], indices[0:n_docs])
        g = self.loadTrainModel()
        test_result = g.test(test_set)
        return test_set, test_result

    def saveTrainModel(self, *args, **kwargs):
        self.grocery.save()
        os.rename(self.PREFIX + self.grocery_name + '_train.svm', self.PREFIX + self.MODEL_DIR + self.grocery_name + '_train.svm')
        return

    def loadTrainModel(self, *args, **kwargs):
        os.rename(self.PREFIX + self.MODEL_DIR + self.grocery_name + '_train.svm', self.PREFIX + self.grocery_name + '_train.svm')
        self.grocery.load()
        os.rename(self.PREFIX + self.grocery_name + '_train.svm', self.PREFIX + self.MODEL_DIR + self.grocery_name + '_train.svm')
        return self.grocery

    def predict(self, line, **kwargs):
        tag = self.grocery.predict(line)
        return tag

    def test(self, *args, **kwargs):
        test_src = str(kwargs["test_src"])
        test_result = self.grocery.test(test_src)
        print "Total Accuracy", test_result

        return test_result
