from pymongo import MongoClient

dbClient = MongoClient("mongodb://127.0.0.1:27017")
db = dbClient["nick"]
collection = db["tickets"]
cursor = collection.find({}, {'body': 1, 'tags': 1, '_id': 0})

f = open("/Users/junipe/Workspace/research/nick/prediction/training_data/training_samples.txt", "wb")
for i in cursor:
    if str(type(i["tags"])) != "<type 'NoneType'>" and str(type(i["body"])) != "<type 'NoneType'>":
        for tag in i["tags"]:
            text = ' '.join(' '.join(filter(None, i["body"].split("\r"))).split("\n"))
            f.write("%s\t%s\n" % (tag.encode("utf-8"), text.encode("utf-8")))
f.close()
