import argparse
import logging
import random
import time
from tqdm.auto import tqdm
import numpy as np
import pandas as pd
import torch
import re
import statistics
import nltk
from scipy import stats
from scipy.spatial.distance import *
from scipy.special import kl_div
from nltk import sent_tokenize
nltk.download('wordnet')
from nltk.wsd import lesk

def jaccard_score(a, b):
    return len(a.intersection(b))/float(len(a.union(b)))

# def new_score(a, b):
#     return (len(a.union(b))-len(a.intersection(b)))/float(len(a.union(b)))


def main():
    random.seed(123)
    np.random.seed(123)
    torch.manual_seed(123)
    blank_out = 0

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default=None, type=str, required=True, help='Data directory')
    parser.add_argument('--results_dir', default=None, type=str, required=True, help='Results directory.')
    parser.add_argument('--top_k',default=2,type=int, required=False, help='topk')
    args = parser.parse_args()
    top_k = args.top_k

    # data_c1 = pd.read_csv('{}/english/corpus1/ccoha1.csv'.format(args.data_dir)).loc[:,['text']]
    # data_c2 = data = pd.read_csv('{}/english/corpus2/ccoha2.csv'.format(args.data_dir)).loc[:,['text']]

    
    # c1_text = []
    # for i in data_c1['text']: c1_text.append(i)
    # c2_text = []
    # for i in data_c2['text']: c2_text.append(i)

    # print(len(c1_text),len(c2_text))

    # c1_text = c1_text[:1000]
    # c2_text = c2_text[:1000]

    

    target_dict1 = {}
    target_dict2 = {}
    target_sen1 = {}
    target_sen2 = {}
    target = pd.read_csv('{}/english/targets.csv'.format(args.data_dir)).loc[:,['text']]
    # target = pd.read_csv('{}/targets.csv'.format(args.data_dir)).loc[:,['text']]
    # target = pd.read_csv('{}/targets-de.csv'.format(args.data_dir)).loc[:,['text']]
    # target = pd.read_csv('{}/targets-sv.csv'.format(args.data_dir)).loc[:,['text']]
    # target = pd.read_csv('{}/targets-la.csv'.format(args.data_dir)).loc[:,['text']]
    target_list = []
    for i in target['text']:
        target_list.append(i)
        target_dict1[i] = {}
        target_dict2[i] = {}
        target_sen1[i] = []
        target_sen2[i] = []

    with open('{}/english/hypothesis/1675688550.ccoha1.lmms.key.txt'.format(args.data_dir), 'r') as f:
    # with open('{}/results/1678706320.ccoha1.ares.key'.format(args.data_dir), 'r') as f:
    # with open('{}/results/1683503899.corpus1-de.ares-multi.key'.format(args.data_dir), 'r') as f:
    # with open('{}/results/1683503861.corpus1-sv.ares-multi.key'.format(args.data_dir), 'r') as f:
    # with open('{}/results/1683503955.corpus1-la.ares-multi.key'.format(args.data_dir), 'r') as f:
        for line in f:
            # print(line)
            # line = line.replace("[formel]", '')
            line = re.split('\[\d+',line)[-1]
            line = line.split('],', 1)[1].lstrip(', ').strip('[]\n')
            word = line.split(', ',1)[0]
            sense = []
            score = []
            sets = line.split(', ',1)[1].strip('[(]').split(', (')
            if len(sets[0].strip('')) <= 0: continue
            for k in range(min(top_k,len(sets))):
                tup = line.split(', ',1)[1].strip('[(]').split(', (')[k]
                sense.append(tup.split(', ')[0].strip("''"))
                score.append(float(tup.split(', ')[1].strip(")")))
                target_sen1[word].append((sense,score))

    with open('{}/english/hypothesis/1675772781.ccoha2.lmms.key.txt'.format(args.data_dir), 'r') as f:
    # with open('{}/results/1678723191.ccoha2.ares.key'.format(args.data_dir), 'r') as f:
    # with open('{}/results/1683503919.corpus2-de.ares-multi.key'.format(args.data_dir), 'r') as f:
    # with open('{}/results/1683503865.corpus2-sv.ares-multi.key'.format(args.data_dir), 'r') as f:
    # with open('{}/results/1683503947.corpus2-la.ares-multi.key'.format(args.data_dir), 'r') as f:
        for line in f:
            # print(line)
            # line = line.replace("[formel]", '')
            line = re.split('\[\d+',line)[-1]
            line = line.split('],', 1)[1].lstrip(', ').strip('[]\n')
            word = line.split(', ',1)[0]
            sense = []
            score = []
            sets = line.split(', ',1)[1].strip('[(]').split(', (')
            if len(sets[0].strip('')) <= 0: continue
            for k in range(min(top_k,len(sets))):
                tup = line.split(', ',1)[1].strip('[(]').split(', (')[k]
                sense.append(tup.split(', ')[0].strip("''"))
                score.append(float(tup.split(', ')[1].strip(")")))
                target_sen2[word].append((sense,score))

    # progress_bar = tqdm(range(len(target_list)))
    for target_word in target_list:
        # print(target_word, len(target_sen1[target_word]), len(target_sen2[target_word]))
        target_sen1[target_word] = random.sample(target_sen1[target_word],min(len(target_sen1[target_word]), len(target_sen2[target_word])))
        target_sen2[target_word] = random.sample(target_sen2[target_word],min(len(target_sen1[target_word]), len(target_sen2[target_word])))
        
        for senses, scores in target_sen1[target_word]:
            for k in range(len(senses)):
                if senses[k] not in target_dict1[target_word]:
                    target_dict1[target_word][senses[k]] = scores[k]/sum(scores)
                else: target_dict1[target_word][senses[k]] += scores[k]/sum(scores)
        for senses, scores in target_sen2[target_word]:
            for k in range(len(senses)):
                if senses[k] not in target_dict2[target_word]:
                    target_dict2[target_word][senses[k]] = scores[k]/sum(scores)
                else: target_dict2[target_word][senses[k]] += scores[k]/sum(scores)
        # progress_bar.update(1)



    dis_dict = {}

    # cnt = 0
    for target_word in target_list:
        sense_set = set(target_dict1[target_word]).union(set(target_dict2[target_word]))
        # cnt += len(sense_set)
        sense_ids1 = []
        sense_ids2 = []
        for sense in sense_set:
            if sense in target_dict1[target_word]:
                sense_ids1.append(target_dict1[target_word][sense])
            else: sense_ids1.append(0.0)
            if sense in target_dict2[target_word]:
                sense_ids2.append(target_dict2[target_word][sense])
            else: sense_ids2.append(0.0)
        if len(sense_ids1) <= 0 and len(sense_ids2) <= 0:
            dis_dict[target_word] = 0
            continue
        # sum1 = sum(sense_ids1)
        # sum2 = sum(sense_ids2)
        # sense_ids1 = [x/sum1 for x in sense_ids1]
        # sense_ids2 = [x/sum2 for x in sense_ids2]
        
        # if 0 in sense_ids1: sense_ids1 = [float(x + 0.00001) for x in sense_ids1]
        # if 0 in sense_ids2: sense_ids2 = [float(x + 0.00001) for x in sense_ids2]

        # dis_dict[target_word] = sum(kl_div(sense_ids1,sense_ids2))
        dis_dict[target_word] = jensenshannon(sense_ids1,sense_ids2)
    # print(cnt/len(target_list))

    # for target_word in target_list:
    #     jaccard_dict[target_word] = new_score(target_dict1[target_word], target_dict2[target_word])

    with open(f'{args.results_dir}/english/hypothesis/senseset_c1_lmms_top{top_k}.txt', 'w') as f:
    # with open(f'{args.results_dir}/results/senseset_c1_ares_top{top_k}.txt', 'w') as f:
        for target_word in target_list:
            f.write(f'{target_word}')
            for sense in target_dict1[target_word]:
                f.write(f',{sense}:{target_dict1[target_word][sense]}')
            f.write('\n')
    with open(f'{args.results_dir}/english/hypothesis/senseset_c2_lmms_top{top_k}.txt', 'w') as f:
    # with open(f'{args.results_dir}/results/senseset_c2_ares_top{top_k}.txt', 'w') as f:
        for target_word in target_list:
            f.write(f'{target_word}')
            for sense in target_dict2[target_word]:
                f.write(f',{sense}:{target_dict2[target_word][sense]}')
            f.write('\n')
    with open(f'{args.results_dir}/english/hypothesis/dis_dict_top{top_k}.txt', 'w') as f:
    # with open(f'{args.results_dir}/results/dis_dict_top{top_k}.txt', 'w') as f:
        for target_word in target_list:
            f.write(f'{target_word},{dis_dict[target_word]}\n')
    
    with open(f'{args.results_dir}/english/hypothesis/score_top{top_k}.txt', 'w') as f:
    # with open(f'{args.results_dir}/results/score_top{top_k}.txt', 'w') as f:
        for target_word in target_list:
            f.write(f'{dis_dict[target_word]},')

    # with open(f'{args.results_dir}/english/hypothesis/new_score.txt', 'w') as f:
    #     for target_word in target_list:
    #         f.write(f'{target_word},{new_dict[target_word]}\n')

    truth = []
    with open('data/english/truth/graded.txt', 'r') as f:
    # with open('test_data_public/english/truth/graded.txt', 'r') as f:
    # with open('test_data_public/german/truth/graded.txt', 'r') as f:
    # with open('test_data_public/swedish/truth/graded.txt', 'r') as f:
    # with open('test_data_public/latin/truth/graded.txt', 'r') as f:
        lines = f.readlines()
        for l in lines:
            truth.append(l.split()[-1])

    blank_set = []
    for index, target_word in enumerate(target_list):
        if max(len(target_dict1[target_word]),len(target_dict2[target_word])) <= 0:
            blank_set.append(index)
    print(blank_set, len(blank_set))

    new = [dis_dict[target_word] for target_word in target_list]

    if blank_out:
        new_ = [new[i] for i in range(len(new)) if i not in blank_set ]
        truth_ = [truth[i] for i in range(len(truth)) if i not in blank_set ]
        new = new_
        truth = truth_

    score = stats.spearmanr(truth,new)[0]
    print(score,truth,new)

if __name__ == '__main__':

    start_time = time.time()

    main()

    print('---------- {:.1f} minutes ----------'.format((time.time() - start_time) / 60))
    print()