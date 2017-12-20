# 就文本聚类来说，长文本（文章），往往用 TF-IDF 就有好效果；
# 太短的话（Twitter），则有 biterm topic models 等特别设计的方法来处理。
# 中等长度的文本（十多个关键词当 feasures），可以使用 LDA 做话题建模，可以增量分析话题走向，并能自动将文本聚类。
# 本工作采用 LDA。

# LDA 的计算复杂度，比起 LSI 和 TF-IDF 都高上许多。
# 使用 LDA 时，需要指定话题建模的数目，在数万个文本，且分析的群数不太多（<100）时，通常数分钟内能完成。
# 我们使用 TF-IDF 的文本向量，使用 K-means 扫过几个可能的 K 值（K = 3 到 100），
# 然后用 silhouette_score 决定最好的 K 值，在用它决定 LDA 的聚类数目。

# 话题建模后，可以看到：
# (1) 不同话题的文本数目；
# (2) 每个话题的关键词分布；
# (3) 每个关键词在不同话题的分布。

# 当我们有了 corpus 和 dictionary 后（初始由 TF-IDF 建立，或自档案读入）
# 才可以做 LDA 话题建模
from nick import ModelBuilder
from nick import DocumentClusterer
mdl = ModelBuilder(method='tfidf')
tfidf = mdl.buildModelAndSave()
tfidf_sim_mat = mdl.buildSimilarityMatrix()
embedding_matrix = mdl.getEmbeddingMatrix(build=True)

# 这里决定 LDA 要分配的话题数目
dc = DocumentClusterer()
# best_K = dc.determineK(embedding_matrix, range_n_clusters_min = 3, range_n_clusters_max = 100)
best_K = 10
# km = dc.fitClusters(embedding_matrix, n_clusters = best_K)
# clusters = dc.getClusters()

# 建立 LDA 模型
# 参数：
# num_topics: LDA 的话题数目
# chunksize: 以多少文本唯一单位，去更新生成中的 LDA 模型
lda = ModelBuilder(method='lda')
lda_model = lda.buildModelAndSave(num_topics=best_K, chunksize=600)

# 因为 LDA 是一个 "软模型"，它并不会将一个工单分配到特定一个话题里，而是在许多话题的分布。
# 如果要问 "哪一种话题的工单" 数目最多？这不是好的问题。
# 但是，我们还是可以在这个分布里，找出权重最大的话题，当做这个工单所属的话题。
# 以下做这件事。
num_topics = {}
for doc in lda.lda.corpus:
    dummy = lda.lda.lda.get_document_topics(doc)
    topic_number = sorted(dummy, key=lambda x: x[1], reverse=True)[0][0]
    try:
        num_topics[topic_number]
    except:
        num_topics[topic_number] = 1
    else:
        num_topics[topic_number] += 1

# 这里可以输出初始的话题分类结果
for i in lda_model.lda.show_topics():
    print 'topic number', i[0], ':', i[1]


# 这里输出一个代表此次 LDA 的 JSON 结果:
lda_result = {}
for topic in lda_model.lda.show_topics():
    lda_result[topic[0]] = {}
    lda_result[topic[0]]["size"] = num_topics[topic[0]]

    dummy_topical_array = map(str.strip, topic[1].encode('utf-8').split('+'))
    lda_result[topic[0]]["topics"] = map(lambda x: [x.split('*')[0], x.split('*')[1].decode("utf-8")], dummy_topical_array)

# save as json
import json
json.dump(lda_result, open("./lda_result.json", "w"), ensure_ascii=True)

# 如果以上程序是用 Python 2 完成，则关键词会存成 unicode，目前还没有解决办法。只得先用 Python 3 重新存一次了。
# run with Python 3 below:
import json
import io
a = json.load(io.open("./lda_result.json", "r", encoding="utf-8"))
json.dump(a, open("./lda_result.json", "w"), ensure_ascii=False )