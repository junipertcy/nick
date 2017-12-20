import prediction

pt = prediction.TagPredictor(
  grocery_name = "test",
  method = "jieba",
  train_src = "./prediction/training_data/training_samples.txt"
)

# 我们也可以使用已分词好的数据，作为 SVM 建模的 features(推荐)
# pt = prediction.TagPredictor(
#   grocery_name = "test",
#   method = "processed",
#   train_src = "./prediction/training_data/training_samples_processesssssssd.txt"
# )

# 这个方法，提供我们在 training data 中，比较两种模型的好坏。
# 第一种喂进全部的 training data 数据，第二种会把 recall rate 较低的 labels 拿走（自己设定 threshold）
# 最后返回第二种模型
pt.autoEvaluation(threshold = 0.2)

# 如果我们决定不用消除掉一些 labels 来训练的模型（决定使用原模型），需要再跑一次：
#model = pt.trainFromDocs()

#我们可以喂给模型一句话，看看他会预测出什么标签（唯一一个最可能的标签）
tag = pt.predict("你好，曲径很好我很赞同。出于好奇心我购买了曲径并使用了一会儿。  其实自己并没有多少需求，所以希望能退款。谢谢。")

#甚至我们可以看看这句话，有没有其他可能的标签，结果已排序（从最可能到最不可能）
single_tag_list = sorted(tag.dec_values, key=tag.dec_values.get, reverse=True)
for single_tag in single_tag_list:
    print single_tag

#将模型存起来
pt.saveTrainModel()

#可以读取以建立好的分类模型，读取的依据是这个模型的 `grocery_name`
#pt.loadTrainModel()

#喂进 training dataset，看看不同标签的 accuracy 和 recall rate，
#藉以决定标签和测试数据的选择好不好。
test_result = pt.test(
  test_src = "./prediction/training_data/training_samples.txt"
)

test_result.show_result()