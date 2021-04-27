import sys
import pickle
from stm import StatisticTextMatching
from stm.similarity import TfidfSimilarity, Bm25Similarity

query_list = ['有没有腹胀?']
resp_list = ['涨?', '腹胀吗?', '您是否存在腹胀？']

stm = StatisticTextMatching(resp_list, 3, stop_words=[])
stm.add_sim_instance([TfidfSimilarity('tfidf'), Bm25Similarity()])
stm.run(query_list)

for i in range(len(stm.recall_res)):
    for l in stm.recall_res[i]:
        print(l)
