from stm.preprocess import remove_stop_words
from stm.utils import validate_stop_words

# STM主类
class StatisticTextMatching():
    def __init__(self, query_list, resp_list, stop_words='simple', tokenizer='char'):
        self.query_list = query_list
        self.resp_list = resp_list
        self.stop_words = stop_words 
        self.tokenizer = tokenizer

        self.preprocess()

    # 预处理：去停用词 + 分词（建词表）
    def preprocess(self):
        # stop_words: []表示不去除停用词；不为空的list表示用户自定义停用词表
        # 如果为str类型，支持以下："cn", "baidu", "hit", "scu" 来自https://github.com/goto456/stopwords, 以及 "simple": 个人统计的，常见标点符号+ 的、得之类的字

        if validate_stop_words(self.stop_words):
            print('loading stop words of {}......'.format(self.stop_words))
            self.query_list = remove_stop_words(self.query_list, self.stop_words)
            self.resp_list = remove_stop_words(self.resp_list, self.stop_words)
            print(self.query_list)
            print(self.resp_list)
        


            

