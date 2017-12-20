from similarity import TokenCoOccur as token_co_o
co_occur = token_co_o()
g = co_occur.constructGraph()


co_occur.dictionary.token2id["产品".decode("utf-8")]
# 196

for i in co_occur.topKTokens(196, 3):
  print i