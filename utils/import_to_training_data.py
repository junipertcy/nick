from pymongo import MongoClient
from nick import KeywordExtractor as keyExt
from tgrocery import Grocery

def _custom_tokenize(line):
    key_ext = keyExt()
    tokens = key_ext.calculateTokens(
        line,
        doc_len_lower_bound = 5,
        doc_len_upper_bound = 500,
        method = "normal"
    )
    return tokens

dbClient = MongoClient("mongodb://127.0.0.1:27017")
db = dbClient["nick"]
collection = db["tickets"]
cursor = collection.find({}, {'body': 1, 'tags': 1, '_id': 0})

f = open("/Users/junipe/Workspace/research/nick/prediction/training_data/training_samples_processed.txt", "wb")
for i in cursor:
    if str(type(i["tags"])) != "<type 'NoneType'>" and str(type(i["body"])) != "<type 'NoneType'>":
        for ind, tag in enumerate(i["tags"]):
            if ind == 0:
                text = ' '.join(' '.join(filter(None, i["body"].split("\r"))).split("\n"))
                try:
                    text = _custom_tokenize(text.encode("utf-8"))
                except Exception as e:
                    if e != KeyboardInterrupt:
                        pass
            if str(type(text)) != "<type 'NoneType'>":
                f.write("%s\t%s\n" % (tag.encode("utf-8"), text.encode("utf-8")))
                f.flush()
f.close()
