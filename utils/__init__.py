import time
class Timer(object):
    def __init__(self, notion):
        self.notion = notion

    def __enter__(self):
        self.start = time.time()

    def __exit__(self, *args):
        self.end = time.time() - self.start
        print '{} executed: {}s'.format(self.notion, self.end)
