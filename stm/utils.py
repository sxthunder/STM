import os

# 验证stop_words是否合法, 不合法则直接raise exception
# 如果使用stopwords则返回True，否则返回False
def validate_stop_words(stop_words):
    if not isinstance(stop_words, (list, str)):
        raise Exception('Stop words must be list or str, the input stop words has type of {}'.format(type(stop_words)))
    
    if isinstance(stop_words, str) and stop_words not in ['baidu', 'cn', 'simple', 'scu', 'hit']:
        raise Exception("If stop words has the type of str, it must be in ['baidu', 'cn', 'simple', 'scu', 'hit']")
    
    return stop_words != []

# 读取默认的stop_words
def load_default_stop_words(stop_words):
    stop_words_list = []
    stop_words_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'stop_words', '{}_stopwords.txt'.format(stop_words))
    with open(stop_words_path, 'r') as f:
        stop_words_list = [x.replace('\n', '') for x in f]
        f.close()
    
    return stop_words_list

# 对于stop words中属于re特殊token的字符进行转移
def update_re_token(stop_words):
    re_token_list = ['*', '.', '?', '|', '^', '$', '+', '(', ')']
    for idx, stop_word in enumerate(stop_words):
        if stop_word in re_token_list:
            stop_words[idx] = "\\" + stop_word
    return stop_words