from ax import optimize
import random
import numpy as np
import math
import torch
from sklearn.metrics import accuracy_score

RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

def cal_acc(p):
    y_pred = []
    for i in dev_score:
        if i >= p['threshold']:
            y_pred.append(1)
        else: y_pred.append(0)
    return accuracy_score(y_pred, y_dev_true)

y_true = [1,0,0,1,0,1,0,0,1,0,0,0,1,1,1,0,1,0,0,0,0,1,1,1,0,1,1,0,0,0,1,0,1,1,0,0,0] #EN
y_dev_index = random.sample(range(0,len(y_true)), k = int(math.ceil(0.2 * (len(y_true)))))
y_dev_true = [y_true[x] for x in y_dev_index]


score = [0.06913546869210442,0.07397745332618848,0.19785146387381133,0.1131797404133528,0.004318754334741524,0.09946588542159614,0.001162332279992383,0.0009074159918919049,0.052143505498438755,0.1496936055832589,0.010804680384721627,0.12028478332809796,0.017409152928720677,0.1904035240164906,0.16955860156149669,0.017616894912445998,0.0,0.007476481336641781,0.01091338787916992,0.12282238632884916,0.06344834554572652,0.30710050860329324,0.009501951600643276,0.013880935529942948,0.007944653448690957,0.08354474015954104,0.2125640932818353,0.00510511813179801,0.056169689654651786,0.04436839826277124,0.05498851171427196,0.09807177855205258,0.005889064534383659,0.09634578062310747,0.019750959081231433,0.11793904006128783,0.07980026760750233,]
dev_score = [score[x] for x in y_dev_index]
# word_ls = ['attack','bag','ball','bit','chairman','circle','contemplation','donkey','edge','face','fiction','gas','graft','head','land','lane','lass','multitude','ounce','part','pin','plane','player','prop','quilt','rag','record','relationship','risk','savage','stab','stroke','thump','tip','tree','twist','word']

bound_score = [x for x in score if x != float('inf') and x != float('-inf') ]
for x in score:
    if x == float('inf'): bound_score.append(2*max(bound_score))
    if x == float('-inf'): bound_score.append(min(bound_score)-abs(min(bound_score)))
print(bound_score)
best_parameters, best_values, _, _ = optimize(
    parameters=[
    {"name": "threshold",
    "type": "range",
    "bounds": [float(min(bound_score)), float(max(bound_score))],},],
    evaluation_function=cal_acc,
    minimize=False,random_seed=RANDOM_SEED)
print(best_parameters,f'best_acc: {cal_acc(best_parameters)}')

y_pred = [1 if x>best_parameters['threshold'] else 0 for x in score]
con = []
for i in range(len(y_pred)):
    if y_pred[i] == 1:
        if y_true[i]: con.append('TP')
        else: con.append('FP')
    else:
        if not y_true[i]: con.append('TN')
        else: con.append('FN')

print(accuracy_score(y_pred, y_true))
print(y_pred)
print(con)
