from pymongo import MongoClient
from collections import defaultdict

frequency = defaultdict(int)

dbClient = MongoClient("mongodb://127.0.0.1:27017")
db = dbClient["nick"]
collection = db["tickets"]
cursor = collection.find({}, {'tags': 1, '_id': 0})

f = open("/Users/junipe/Workspace/research/nick/model_building/user_dict/user_dict_from_tags.txt", "wb")
for i in cursor:
    if str(type(i["tags"])) != "<type 'NoneType'>":
        for tag in i["tags"]:
            frequency[tag.encode("utf-8").lower()] += 1
            if frequency[tag.encode("utf-8").lower()] == 1:
                f.write("%s n\n" % (tag.encode("utf-8")))
f.close()
