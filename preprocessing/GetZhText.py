import re

class GetZhText(object):

    def __init__(self):
        pass

    def __call__(self, doc):
        regex = []
        regex += [ur'[\u4e00-\ufaff]']
        regex = "|".join(regex)
        r = re.compile(regex)
        return ''.join(r.findall(doc))
