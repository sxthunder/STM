# 预处理：去除停用词+分词
import re
from stm.utils import load_default_stop_words, update_re_token

def remove_stop_words(input_type, query_list, resp_list, stop_words):
    def re_subn(pattern, l):
        nl = []
        for s in l:
            nl.append(re.subn(stop_words_re, "", s)[0].replace(' ', ''))
        return nl
    
    def token_subn(stop_words, l):
        nl = [[x for x in ll if x not in stop_words] for ll in l]
        return nl

    if isinstance(stop_words, str):
        stop_words = load_default_stop_words(stop_words)

    # input_type为sen时，直接使用正则匹配
    if input_type == 'sen':
        # 对于stop words中属于正则特殊字符的，进行转义处理
        stop_words = update_re_token(stop_words)
        
        stop_words_re = '|'.join(stop_words)
        new_query_list = re_subn(stop_words_re, query_list)
        new_resp_list = re_subn(stop_words_re, resp_list)

    # input_type为token时，直接判断是否在stop_words中，然后替换
    else:
        new_query_list = token_subn(stop_words, query_list)
        new_resp_list = token_subn(stop_words, resp_list)
        
    return new_query_list, new_resp_list



    





