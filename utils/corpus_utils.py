#not used...
from gensim import corpora

class CorpusIterator(object):

    def __init__(self, *args, **kwargs):
        self.kwargs = kwargs

    def __call__(self, x):
        raise NotImplementedError

    def setIteratorFunction(self, iter_func, *args):
        self.iter_func = iter_func
        self.iter_args = args

    def __iter__(self):
        for x in self.iter_func(*self.iter_args):
            yield x

    def tokenizedTextIterator(self):
        pass