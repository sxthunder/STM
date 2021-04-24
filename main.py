from stm import StatisticTextMatching

# query_list =['大家好|,我是单位房的的的的的*,。，水电费？水电费水电费%']
query_list = [['大家好', '的', '的', '*']]
stm = StatisticTextMatching(query_list, [], 'token', 'simple')


