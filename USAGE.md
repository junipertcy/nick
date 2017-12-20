一个正常的 NLP 管线顺序包括：

1. 关键词提取，同时建立 a) 关键词字典；b) 文本缩略表；c) 文本嵌入矩阵。以上三者都可以增量计算，并依照工单类型，需要长期维护。
2. 工单聚类，同时建立 d) 工单聚类模型。此模型用以对已知与未知工单进行分类，不能增量计算，但定期重新计算一次即可。
3. 关键词相似度。
4. 自动打标签，同时建立 e)标签预测模型。

如果想要支持多语系，必须有多语系的 postagger 库数据。Stanford CoreNLP 包支持了英语、中文及德法语的分词，目前系统默认不使用 CoreNLP。外国语言的分词工具，请见 [http://nlp.stanford.edu/software/tagger.shtml]

关键词提取的接口主要是 `KeywordExtractor` 里面的 `getKeywordsAndSave`，请见 [example_keyword_extraction.py](example_keyword_extraction.py) 的范例和说明。

工单聚类的部份，在文本分词后，采用 gensim 里的 LDA 建模；并使用 Python 工具包 LDAvis，直接输出成带有 D3 交互的 html，作为前端展示页面。使用 LDA 时的重要参数：(1) 聚类数; (2) 一次性训练的文本数量. (1) 我们采用 clustering/DetK.py 里的 silhouette score 决定，给定 K 的扫描范围，输出最佳的 K 值。(2) 取决于工单总数，一般设为工单总数的 10%.

目前计算关键词相似度的方式，是用 “建 Graph” 来处理。即，出现在同一工单的关键词都连上一条边，遍历所有工单，最后这个关键词网络中，每个节点所连出的边就能排序，选出最直接相关的其他关键词。请安装 Python graph-tool 这个 library。这个库很重，许多核心功能是在 C++ 完成的，需要在机器上 compile 才能使用，安装需要一到两个小时。目前用它来算相似度，可能牛刀小用，但是以后会用得更多的。

自动打标签的功能，核心采用 TextGrocery 这个 Python library 来建构 SVM 分类模型。以初次开发的测试数据来说，recall rate 大约 70%. 这之中，经过包装后，支援两种特徵抓取模式: (1) 内建的 Jieba 断词; (2) 用户字定义的断词。建议采用 (1) 即可，因为保持适当的杂讯在特徵数据中，可以帮助文本分类。