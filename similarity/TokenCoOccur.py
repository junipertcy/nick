from gensim import corpora
import route.config as conf

class TokenCoOccur(object):

    def __init__(self):
        self.conf_similarity = conf.load("similarity")
        self.dictionary = corpora.Dictionary.load(self.conf_similarity["prefix"]+self.conf_similarity["dictionary"])
        self.corp = corpora.MmCorpus(self.conf_similarity["prefix"]+self.conf_similarity["corpus"])
        pass

    def constructGraph(self, *args, **kwargs):
        import graph_tool.all as gt
        self.g = gt.Graph()
        self.g.set_directed(False)
        self.g.vertex_properties["token_name"] = self.g.new_vertex_property("string")
        self.g.edge_properties["edge_weight"] = self.g.new_edge_property("int")

        # iterating all tokens in documents
        # takes some time ...
        for doc in self.corp:
            for ind_i, source in enumerate(doc):
                for ind_j, target in enumerate(doc):
                    if ind_i < ind_j:
                        self.g.add_edge(source[0], target[0])
                        try:
                            self.g.ep["edge_weight"][(source[0], target[0])]
                        except:
                            self.g.ep["edge_weight"][(source[0], target[0])] = 1
                        else:
                            self.g.ep["edge_weight"][(source[0], target[0])] += 1

        for node in self.dictionary.id2token.items():
            self.g.vp["token_name"][node[0]] = node[1]

        return self.g

    def topKTokens(self, token_node, top_k):
        edge_w = {}
        for i in self.g.vertex(token_node).out_edges():
            edge_w[(i.source().__int__(), i.target().__int__())] = self.g.ep["edge_weight"][i]

        top_k_list = sorted(edge_w.items(), key=lambda x: x[1], reverse=True)[:top_k]
        top_k_ids = [i[0][1] for i in top_k_list]
        top_k_tokens = []
        for _id in top_k_ids:
            top_k_tokens.append(self.dictionary.id2token[_id])

        return top_k_tokens