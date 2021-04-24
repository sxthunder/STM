# 预处理：去除停用词+分词
import re
from stm.utils import load_default_stop_words, update_re_token

def remove_stop_words(l, stop_words):
    if isinstance(stop_words, str):
        stop_words = load_default_stop_words(stop_words)
    
    # 对于stop words中属于正则特殊字符的，进行转义处理
    stop_words = update_re_token(stop_words)
    
    stop_words_re = '|'.join(stop_words)
    nl = []
    for s in l:
        nl.append(re.subn(stop_words_re, "", s)[0].replace(' ', ''))
    return nl



    





