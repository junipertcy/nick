from nick import KeywordExtractor
from nick import ModelBuilder

keyExt = KeywordExtractor()

# OR, 可以自定义不需要提取关键词的工单
keyExt = KeywordExtractor(excluded_docs = [
  "提交过退款申请后，我们会逐一处理，退款将在我们回复你之后尽快完成，退款的退回路径是原路返回，请耐心等待。",
  "提交过退款申请后，我们会逐一处理，退款将在我们回复你之后尽快完成，请耐心等待。"
])

num = 10000


# 将数据库数据存成 .doclist 格式（其实就是一行一个工单）
# 若不用 *.doclist 档案作为中介档案，则在 getKeywordsAndSave 方法中，
# 使用 static_file = False 即可。
keyExt.saveToDoclist(num, separated=False)

# 以下使用 pynlpir 提取关键词。
# freq_lower_bound( Int: 10 )
#
# 在全局内，某个特定单词（两个字元以上）出现的次数，必须要大于这个特定的值。
# 这会取决于一次喂入的数据量，如果数据量大，自然可选择大一点的数字，而不会过滤有效信息。
# 实验中，喂入长短不一的 10000 条工单，选择了 10 当做阈值，有不错的结果。

# token_len_lower_bound( Int: 1)
#关键词的字元长度限制。默认就是 1 了，如果这么做的话，像 "贵" 这样的词不会出现，
#但是可以有效避免中文分词产生的杂讯。
#
# keyExt.getKeywordsAndSave(
#   static_file = True,
#   source_name = 'num_docs-10000.doclist',
#   target_name = "test",
#   method = "normal",
#   freq_lower_bound = 10,
#   token_len_lower_bound = 1,
#   doc_len_lower_bound = 5,
#   doc_len_upper_bound = 500
# )

keyExt.getKeywordsAndSave(
  static_file = True,
  source_name = 'num_docs-{}.doclist'.format(num),
  target_name = "test",
  method = "normal",
  freq_lower_bound = 10,
  token_len_lower_bound = 1,
  doc_len_lower_bound = 5,
  doc_len_upper_bound = 500
)

#doc_list = ['以下是提取关键词的工具', '记得关键词要以Python list 方式喂进去', '之后，你就会拿到原始关键词数据了',
#'但是  如果要有更好的关键词，最好在话题建模后，使用 refineKeywords 方法', '这样得到的关键词会更好']
#keyExt.getKeywordsAndSave(
#  doc_list,
#  static_file = False,
#  target_name = "test",
#  method = "normal",
#  freq_lower_bound = 0,
#  token_len_lower_bound = 1,
#  doc_len_lower_bound = 5,
#  doc_len_upper_bound = 500
#)

#以上，完成初步的关键词提取。

#接下来，你需要建立 TF-IDF 的模型，一来作为文本聚类、二来作为进一步过滤关键词所用：
mdl = ModelBuilder(method='tfidf')
tfidf = mdl.buildModelAndSave()
tfidf_sim_mat = mdl.buildSimilarityMatrix()
embedding_matrix = mdl.getEmbeddingMatrix(build=True)

#最后拿到经过 TF-IDF 过滤以后的结果
keyExt.refineKeywords(method='tfidf', target_name='test', top_k = 10)


#此方法暫告放棄（deprecated）
#keyExt.getKeywordsAndSave(static_file = True, source_name = 'num_docs-10000.doclist', target_name = "test", method="keyword", freq_lower_bound=1, token_len_lower_bound=1)
#或
# keyExt.getKeywordsAndSave(doc_list, static_file = False, target_name = "test", method="keyword")


