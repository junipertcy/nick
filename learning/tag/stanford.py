from __future__ import print_function
import sys

from threading import Thread
try:
    from Queue import Queue, Empty
except ImportError:
    from queue import Queue, Empty

from nltk.internals import java, _java_options, config_java
from nltk.tag.stanford import StanfordTagger


def _enqueue_output(out, q):
    for line in iter(out.readline, b''):
        q.put(line)
    out.close()

class StanfordTaggerStdio(StanfordTagger):
    def __init__(self, *args, **kwargs):
        super(StanfordTaggerStdio, self).__init__(*args, **kwargs)
        self._thread = None
        self._child = None
        self._queue = None

    def tag(self, tokens):
        _input = ' '.join(tokens).replace('\n', ' ').replace('\r', ' ').replace('\r\n', ' ').strip()
        if len(_input) == 0:
            return []
        # Create pipe if not already opened
        if not self._thread:
            encoding = self._encoding
            default_options = ' '.join(_java_options)
            config_java(options=self.java_options, verbose=False)

            cmd = list(self._cmd)
            cmd.extend(['-encoding', encoding])

            self._child = java(cmd, classpath=self._stanford_jar,
                    stdin='pipe', stdout='pipe', stderr='pipe', blocking=False)
            self._queue = Queue()
            self._thread = Thread(target=_enqueue_output, args=(self._child.stdout, self._queue))
            self._thread.daemon = True
            self._thread.start()

        # clear all newlines, only append one at last for java
        _input += '\n'
        self._child.stdin.write(_input.encode('utf-8'))
        self._child.stdin.flush()
        try:
            return self.parse_output(self._queue.get(timeout=120)) # wait for 2m, usually should return in less than 100ms
        except Empty:
            print('stanford postagger timeout, return empty tuple instead', file=sys.stderr)
            return []

    def parse_output(self, text):
        return [tuple(x.strip().split(self._SEPARATOR)) for x in text.decode('utf-8').strip().split(' ')]


class StanfordPOSTagger(StanfordTaggerStdio):
    _SEPARATOR = '_'
    _JAR = 'stanford-postagger.jar'

    def __init__(self, *args, **kwargs):
        super(StanfordPOSTagger, self).__init__(*args, **kwargs)

    @property
    def _cmd(self):
        return ['edu.stanford.nlp.tagger.maxent.MaxentTagger',
                '-model', self._stanford_model, '-tokenize', 'false','-outputFormatOptions', 'keepEmptySentences']

if __name__ == '__main__':
    tagger = StanfordPOSTagger('../../stanford-postagger/models/english-left3words-distsim.tagger',
            path_to_jar='../../stanford-postagger/stanford-postagger-3.6.0.jar')
    print(tagger.tag(['this', 'is', 'one']))
    print(tagger.tag(['this', 'is', 'two']))
    print(tagger.tag(['this', 'is', 'three']))
    print(tagger.tag([]))
    print(tagger.tag([u'p.pac', u'http', u'ebyx', u'url', u'p.getqujing.com', u'wifi', u'youtube', u'facebook', u'iphone', u'wi-fi', u'pants']))
