import re

class GetEnTokens(object):

    def __init__(self):
        regex = []
        #regex += ['^wi$']
        regex += ['[A-Za-z\-\.A-Za-z]+']
        regex = "|".join(regex)
        self.r = re.compile(regex)

    def __call__(self, doc):
        return self.r.findall(doc)

if __name__ == '__main__':
    import sys
    g = GetEnTokens()
    for line in sys.stdin:
        print ' '.join(g(line))
