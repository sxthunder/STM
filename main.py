import sys
import pickle
from stm import StatisticTextMatching
from stm.similarity import TfidfSimilarity

project_path = '/home/liangming/nas/ml_project/medical_dual'
sys.path.append(project_path)

fade_train = pickle.load(open('{}/fade_train'.format(project_path), 'rb'))
test = pickle.load(open('{}/PhasesB/PhasesBTestData.pk'.format(project_path), 'rb'))
train = pickle.load(open('{}/MedDG_share/new_train.pk'.format(project_path), 'rb'))

# 医生所有的回复
resp_list = set()
for dialogue in train:
    for sen in dialogue:
        if sen['id'] == 'Doctor':
            resp_list.add(sen['Sentence'])
resp_list = list(resp_list)
ans = pickle.load(open('{}/xuekui_final_ans'.format(project_path), 'rb'))
ans = [x for x in ans if x is not None]

query_list = ['有没有腹胀?']
resp_list = ['涨?', '腹胀吗?', '您是否存在腹胀？']
stm = StatisticTextMatching(query_list, resp_list, 3, stop_words=[])
stm.add_sim_instance([TfidfSimilarity()])

stm.run(query_list)
for l in stm.recall_res[0]:
    print(l)

# pickle.dump(stm.recall_res[0], open('./recall_reply', 'wb'))