import networkx as nx
import bm25s
import numpy as np

def GRADA(contents = None, query = None, defence_method = None, alpha = None):

    if defence_method == "no":
        contents =  contents[0:5]

    elif defence_method == "HRSIM":
        bm25_retriever_question = bm25s.BM25(corpus=query)
        bm25_retriever_question.index(bm25s.tokenize(query))
        sim_mat = np.zeros([len(contents), len(contents)])

        for x in range(len(contents)):
            bm25_retriever_x = bm25s.BM25(corpus=contents[x])
            bm25_retriever_x.index(bm25s.tokenize(contents[x]))
            r, scores_q_x = bm25_retriever_question.retrieve(bm25s.tokenize(contents[x]), k=1)
            
            for y in range(len(contents)):
                if x != y and x < y:
                    bm25_retriever_y = bm25s.BM25(corpus=contents[y])
                    bm25_retriever_y.index(bm25s.tokenize(contents[y]))
                    r, scores_x_y = bm25_retriever_y.retrieve(bm25s.tokenize(contents[x]), k=1)
                    r, scores_y_x = bm25_retriever_x.retrieve(bm25s.tokenize(contents[y]), k=1)
                    r, scores_q_y = bm25_retriever_question.retrieve(bm25s.tokenize(contents[y]),
                                                                    k=1)
                    sim_mat[x][y] = 0.5 * scores_x_y[0][0] + 0.5 * scores_y_x[0][0] - alpha * (
                                scores_q_x[0][0] + scores_q_y[0][0])
                    if sim_mat[x][y] <= 0:
                        sim_mat[x][y] = 0.0
                    sim_mat[y][x] = sim_mat[x][y]

        nx_graph = nx.from_numpy_array(sim_mat)
        scores = nx.pagerank(nx_graph)

    elif defence_method == "BM25":
        # BM25
        for x in range(len(contents)):
            bm25_retriever = bm25s.BM25(corpus=contents[x])
            bm25_retriever.index(bm25s.tokenize(contents[x]))
            for y in range(len(contents)):
                if x != y:
                    r, score = bm25_retriever.retrieve(bm25s.tokenize(contents[y]), k=1)
                    sim_mat[x][y] += 0.5 * score[0][0]
                    sim_mat[y][x] += 0.5 * score[0][0]
                    if sim_mat[x][y] <= 0:
                        sim_mat[x][y] = 0.0
        nx_graph = nx.from_numpy_array(sim_mat)
        scores = nx.pagerank(nx_graph)