import os

cwd = os.getcwd()
def read_stop_words(name):
    stop_words = []
    with open('{}/{}_stopwords.txt'.format(cwd, name), 'r') as f:
        stop_words = [x.replace('\n', '') for x in f]
        f.close()
    return stop_words

cn = read_stop_words('cn')
baidu = read_stop_words('baidu')
hit = read_stop_words('hit')
scu = read_stop_words('scu')

total = cn + baidu + hit + scu
total = list(set(total))

simple_stop_words =  [x for x in total if len(x) == 1]

with open('{}/simple_stopwords.txt'.format(cwd), 'w') as f:
    for stop_word in simple_stop_words:
        f.write(stop_word + '\n')
    f.close()
    


