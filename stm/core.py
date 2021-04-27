from stm.preprocess import remove_stop_words 
from stm.utils import validate_stop_words, validate_input_type, validate_tokenizer, validate_vocab_source

# STM主类
class StatisticTextMatching():
    def __init__(self, 
                # query_list, 
                resp_list, 
                k,
                input_type='sen', 
                stop_words='simple', 
                ):         
        # self.query_list = query_list
        self.input_type = input_type
        self.resp_list = resp_list
        self.stop_words = stop_words 
        self.k = k
        self._sim_tensor = []
        self.recall_res = []

        self.resp_list = self.preprocess(self.resp_list)

    # 三维tensor: sim_instance个数 * len(query_list) * len(resp_list)
    @property 
    def sim_tensor(self):
        return self._sim_tensor
    
    def update_sim_tensor(self, sim_matrix):
        self._sim_tensor.append(sim_matrix)

    # 预处理：去停用词 + 分词（建词表）
    def preprocess(self, l):
        print('proprecessing...')
        # 验证input_type
        validate_input_type(self.input_type, l)
            
        # stop_words: []表示不去除停用词；不为空的list表示用户自定义停用词表
        # 如果为str类型，支持以下："cn", "baidu", "hit", "scu" 来自https://github.com/goto456/stopwords, 以及 "simple": 个人统计的，常见标点符号+ 的、得之类的字

        if validate_stop_words(self.stop_words):
            print('loading stop words of {}......'.format(self.stop_words))
            l = remove_stop_words(self.input_type, l, self.stop_words)
        
        return l

    # 添加similarity实例类list
    def add_sim_instance(self, sim_instances):
        self.sim_instances = sim_instances

        for sim_instance in self.sim_instances:
            sim_instance.init_from_stm(self.input_type, self.resp_list)

    def run(self, query_list):
        query_list = self.preprocess(query_list)

        for sim_instance in self.sim_instances:
            # self.update_sim_tensor(sim_instance.run(self.k))
            self.recall_res.append(sim_instance.run(query_list, self.k))





        


            

