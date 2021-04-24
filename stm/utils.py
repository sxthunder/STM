import jieba
import os
from inspect import isfunction

# 验证input_type, input_type表示上传的query和resp是句子级别还是token级别
def validate_input_type(input_type, query_list, resp_list):
    def validate_list_type(l):
        type_dict ={'sen':str, 'token':list}

        for item in l:
            if not isinstance(item, type_dict[input_type]):
                raise Exception('input type is {} but the element in input list is not {}!'.format(input_type, input_type))

    if input_type not in ['sen', 'token']:
        raise Exception('input type must be "sen" or "token"')
    
    validate_list_type(query_list)
    validate_list_type(resp_list)

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

# 验证tokenizer是否合法,如果合法则返回tokenizer方法，否则raise exception
# 要么为str:['jieba', 'char']
# 要么为自定义的一个方法：输入一个str，返回分词后的token list
def validate_tokenizer(tokenizer, dict_path):
    new_t = None
    if tokenizer not in ['jieba', 'char'] and isfunction(tokenizer):
        raise Exception('tokenizer either be in ["jieba", "char"] or a function')
    
    if tokenizer == 'jieba':
        jieba.load_userdict(open(dict_path))
        new_t = jieba.lcut()
    
    elif tokenizer == 'char':
        new_t = lambda x:list(x)
    
    else:
        new_t = tokenizer
    
    return new_t

# 验证vocab_source是否正确，query resp 或者 both
# 如果符合则返回vocab source list
def validate_vocab_source(query_list, resp_list, vocab_source):
    if vocab_source not in ['query', 'resp', 'both']:
        raise Exception('vocab source either be in ["query", "resp", "both"]')
    
    if vocab_source == 'query':
        return query_list
    elif vocab_source == 'resp':
        return resp_list
    else:
        return query_list + resp_list