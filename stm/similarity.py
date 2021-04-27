import numpy as np
import faiss
from tqdm import tqdm
from stm.utils import validate_tokenizer, validate_vocab_source

# 所有similarity的父类
# 不论什么simialrity方法,输入query_list和resp_list, 根据参数决定的是否需要tokenize, 最终计算出一个len(query_list) * len(resp_list)的一个相似度矩阵
class BasicSimilarity:
    @property
    def input_type(self):
        return self._input_type

    @input_type.setter
    def input_type(self, input_type):
        self._input_type = input_type
    @property
    def query_list(self):
        return self._query_list

    @query_list.setter
    def query_list(self, query_list):
        self._query_list = query_list

    @property
    def resp_list(self):
        return self._resp_list

    @resp_list.setter
    def resp_list(self, resp_list):
        self._resp_list = resp_list

    # 从stm中更新一些参数
    def init_from_stm(self, input_type, resp_list):
        self.input_type = input_type
        self.resp_list = resp_list

    # 计算相似度,返回一个len(query_list), len(resp_list)的matarix
    def run(self):
        raise NotImplementedError

# 计算前需要tokenize的similarity基础类型
class TokenizeSimilarity(BasicSimilarity):
    def __init__(self, name, tokenizer='char', dict_path=None, min_token_freq=0, max_token_freq=1e10):
        print('Initialize {}......'.format(name))
        self.tokenizer = tokenizer
        self.dict_path = dict_path
        self.min_token_freq = min_token_freq
        self.max_token_freq = max_token_freq

    def init_from_stm(self, input_type, resp_list):
        super(TokenizeSimilarity, self).init_from_stm(input_type, resp_list)

        self.build_vocab()
        self._resp_tf_feature = self.tokenize(self.resp_list)

        # resp中加入idf,同时对length做惩罚
        self.idf_vector = np.array([self._vocab_idf[self._reverse_vocab[idx]] for idx in range(self._vocab_size)])
        self.resp_len_vector = np.sum(self._resp_tf_feature, axis=-1)

    def build_vocab(self):
        print('building vocabulary ......')
        # 分词, input_type为sen时才分词
        self.tokenizer = validate_tokenizer(self.tokenizer, self.dict_path, self.input_type)

        self.resp_length = len(self.resp_list)
        self.resp_avg_len = sum([len(x) for x in self.resp_list]) / self.resp_length

        # 先统计词频和idf,进行过滤后加入vocab
        token_count = {}
        idf_count = {}

        for line in self.resp_list:
            tokenize_list = self.tokenizer(line)
            visited_token = set()
            for token in tokenize_list:
                if token not in visited_token:
                    idf_count[token] = idf_count.get(token, 0) + 1
                    visited_token.add(token)
                token_count[token] = token_count.get(token, 0) + 1
        
        self._vocab = {}
        self._vocab_token_count = {}
        self._vocab_idf = {}

        for k, v in token_count.items():
            if v >= self.min_token_freq and v <= self.max_token_freq:
                self._vocab[k] = len(self._vocab)
                self._vocab_token_count[k] = v
                idf = idf_count[k]
                idf = idf + 1 if idf == 0 else idf
                self._vocab_idf[k] = np.log(self.resp_length / (idf))
        
        self._reverse_vocab = {v:k for k, v in self._vocab.items()}
        self._vocab_size = len(self._vocab)

    # l为sen-list, 对l中的每个句子进行tokenize, 其实对应sklearn中tfidfvectorize的transform方法
    def tokenize(self, sen_list):
        print("tokenizeing......")
        # tokenize本质是将l转换成一个shape为(len(l), vocab_size)的矩阵,其中[i][j]表示第i个句子中,vocab内第j个词出现的次数
        feature_matrix = []
        for sen in tqdm(sen_list):
            sen_feature = [0 for _ in range(self._vocab_size)]
            for token in self.tokenizer(sen):
                if token in self._vocab:
                    sen_feature[self._vocab[token]] += 1
            feature_matrix.append(sen_feature)

        feature_matrix = np.array(feature_matrix) 
        return feature_matrix

    def search_by_faiss(self, vocab_size, k, resp_feature, resp_list, query_feature, query_list):
        # 对于这种vector类型的,默认采用faiss;如果user需要得到similarity matrix,则另做调用
        self.index = faiss.IndexFlatIP(vocab_size)
        resp_feature = resp_feature.astype(np.float32)
        self.index.add(resp_feature)
        sim_score_matrix, sim_index_matrix = self.index.search(query_feature, k)

        res = []
        for query_idx, (sim_score, sim_index) in enumerate(zip(sim_score_matrix, sim_index_matrix)):
            res.append([query_list[query_idx]] + [(resp_list[idx], score) for idx, score in zip(sim_index, sim_score)])
        return res

    def get_query_feature(self, query_list):
        self.query_list = query_list
        query_tf_feature = self.tokenize(self.query_list)

        query_tf_feature = (query_tf_feature != 0).astype(np.float32)

        return query_tf_feature
        
class TfidfSimilarity(TokenizeSimilarity):
    # query 和 resp的相似度计算为 : \sum w_tf_in_q * w_tf_in_r * w_idf * sqrt(len(r))
    # w是q和r中共线的词, 如果cal_query_tf = False, 则w_tf_in_q = 1
    # k为召回的个数
    def run(self, query_list, k):
        self._query_bow_feature = self.get_query_feature(query_list)

        # resp中加入idf,同时对length做惩罚
        self._resp_tfidf_feature = np.sqrt(self._resp_tf_feature) * np.expand_dims(self.idf_vector, axis=0) * (1 / np.expand_dims(self.resp_len_vector, axis=-1))

        return self.search_by_faiss(self._vocab_size, k, self._resp_tfidf_feature, self.resp_list, self._query_bow_feature, self.query_list)


class Bm25Similarity(TokenizeSimilarity):
    def __init__(self, k=2.5, b=0.82, name='bm25', tokenizer='char', dict_path=None, min_token_freq=0, max_token_freq=1e10):
        super(Bm25Similarity, self).__init__(name, tokenizer, dict_path, min_token_freq, max_token_freq)
        self.k = k
        self.b = b
    
    def run(self, query_list, k):
        self._query_bow_feature = self.get_query_feature(query_list)

        # 调整后的tf feature 
        # tf = ((k + 1) * tf) / (k(1 - b + b * l) + tf)
        # 其中l为当前doc的length和avg doc length的比值
        self._resp_ad_tf_feature = ((self.k + 1) * self._resp_tf_feature) / (self.k * (1 - self.b + self.b * (np.expand_dims(self.resp_len_vector, axis=1) / self.resp_avg_len))+ self._resp_tf_feature)
        self._resp_bm25_feature = self._resp_ad_tf_feature * np.expand_dims(self.idf_vector, axis=0)

        return self.search_by_faiss(self._vocab_size, k, self._resp_bm25_feature, self.resp_list, self._query_bow_feature, self.query_list)

        
    