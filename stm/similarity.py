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
    def init_from_stm(self, input_type, query_list, resp_list):
        self.input_type = input_type
        self.query_list = query_list
        self.resp_list = resp_list
        # print(self)
        # print(self.input_type, self.query_list, self.resp_list)

    # 计算相似度,返回一个len(query_list), len(resp_list)的matarix
    def run(self):
        raise NotImplementedError

# 计算前需要tokenize的similarity基础类型
class TokenizeSimilarity(BasicSimilarity):
    def __init__(self, name, tokenizer='char', dict_path=None, vocab_source='resp', min_token_freq=0, max_token_freq=1e10, cal_query_tf=False):
        print('Initialize {}......'.format(name))
        self.tokenizer = tokenizer
        self.dict_path = dict_path
        self.vocab_source = vocab_source
        self.min_token_freq = min_token_freq
        self.max_token_freq = max_token_freq
        self.cal_query_tf = cal_query_tf # 是否考虑query的tf

    def init_from_stm(self, input_type, query_list, resp_list):
        super(TokenizeSimilarity, self).init_from_stm(input_type, query_list, resp_list)

        self.build_vocab()
        self._query_tf_feature = self.tokenize(self.query_list)
        self._resp_tf_feature = self.tokenize(self.resp_list)

    def build_vocab(self):
        print('building vocabulary ......')
        # 分词, input_type为sen时才分词
        self.tokenizer = validate_tokenizer(self.tokenizer, self.dict_path, self.input_type)

        vocab_source_list = validate_vocab_source(self.query_list, self.resp_list, self.vocab_source)
        vocab_source_length = len(vocab_source_list)

        # 先统计词频和idf,进行过滤后加入vocab
        token_count = {}
        idf_count = {}

        for line in vocab_source_list:
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
                self._vocab_idf[k] = np.log(vocab_source_length / (idf))
        
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
        
class TfidfSimilarity(TokenizeSimilarity):
    # query 和 resp的相似度计算为 : \sum w_tf_in_q * w_tf_in_r * w_idf * sqrt(len(r))
    # w是q和r中共线的词, 如果cal_query_tf = False, 则w_tf_in_q = 1
    # k为召回的个数
    def run(self, k):
        print('run tf_idf')
        # 如果不需要考虑query的tf,那么直接将其变成01
        if not self.cal_query_tf:
            self._query_tf_feature = (self._query_tf_feature != 0).astype(np.float32)

        self._query_tf_feature = np.sqrt(self._query_tf_feature)

        # resp中加入idf,同时对length做惩罚
        idf_vector = np.array([self._vocab_idf[self._reverse_vocab[idx]] for idx in range(self._vocab_size)])
        resp_len_vector = np.sum(self._resp_tf_feature, axis=-1)
        self._resp_tfidf_feature = np.sqrt(self._resp_tf_feature) * np.expand_dims(idf_vector, axis=0) * (1 / np.expand_dims(resp_len_vector, axis=-1)) 
        self._resp_tfidf_feature = self._resp_tfidf_feature.astype(np.float32)

        # 对于这种vector类型的,默认采用faiss;如果user需要得到similarity matrix,则另做调用
        self.index = faiss.IndexFlatIP(self._vocab_size)
        self.index.add(self._resp_tfidf_feature)
        sim_score_matrix, sim_index_matrix = self.index.search(self._query_tf_feature, k)

        res = []
        for query_idx, (sim_score, sim_index) in enumerate(zip(sim_score_matrix, sim_index_matrix)):
            res.append([self._query_list[query_idx]] + [(self._resp_list[idx], score) for idx, score in zip(sim_index, sim_score)])
        return res

class Bm25Similarity(TokenizeSimilarity):
    def run(self, k, b, name, tokenizer='char', dict_path=None, vocab_source='resp', min_token_freq=0, max_token_freq=1e10, cal_query_tf=False):
        super(Bm25Similarity, self).__init__(name, tokenizer, dict_path, vocab_source, min_token_freq, max_token_freq, cal_query_tf)
        self.k = k
        self.b = b

    def build_vocab(self):
        print('building vocabulary ......')
        # 分词, input_type为sen时才分词
        self.tokenizer = validate_tokenizer(self.tokenizer, self.dict_path, self.input_type)

        vocab_source_list = validate_vocab_source(self.query_list, self.resp_list, self.vocab_source)
        vocab_source_length = len(vocab_source_list)
        self.resp_avg_length = sum([len(x) for x in vocab_source_list]) / vocab_source_length

        # 先统计词频和idf,进行过滤后加入vocab
        token_count = {}
        idf_count = {}

        for line in vocab_source_list:
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
                self._vocab_idf[k] = np.log(vocab_source_length / (idf))
        
        self._reverse_vocab = {v:k for k, v in self._vocab.items()}
        self._vocab_size = len(self._vocab)
