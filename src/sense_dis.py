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

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default=None, type=str, required=True, help='Data directory')
    parser.add_argument('--results_dir', default=None, type=str, required=True, help='Results directory.')
    args = parser.parse_args()


    data_c1 = pd.read_csv('{}/english/corpus1/ccoha1.csv'.format(args.data_dir)).loc[:,['text']]
    data_c2 = data = pd.read_csv('{}/english/corpus2/ccoha2.csv'.format(args.data_dir)).loc[:,['text']]

    c1_text = []
    for i in data_c1['text']: c1_text.append(i)
    c2_text = []
    for i in data_c2['text']: c2_text.append(i)

    print(len(c1_text),len(c2_text))

    # c1_text = c1_text[:1000]
    # c2_text = c2_text[:1000]

    

    target_dict1 = {}
    target_dict2 = {}
    sent_ls1 = {}
    sent_ls2 = {}
    target = pd.read_csv('{}/english/targets.csv'.format(args.data_dir)).loc[:,['text']]
    target_list = []
    for i in target['text']:
        target_list.append(i)
        target_dict1[i] = {}
        target_dict2[i] = {}
        sent_ls1[i] = []
        sent_ls2[i] = []
    
    progress_bar = tqdm(range(len(target_list)))
    for target_word in target_list:
        for sent in c1_text:
            if target_word in sent.split():
                sent_ls1[target_word].append(sent)
        for sent in c2_text:
            if target_word in sent.split():
                sent_ls2[target_word].append(sent)
        progress_bar.update(1)

        print(target_word, len(sent_ls1[target_word]), len(sent_ls2[target_word]))
        sent_ls1[target_word] = random.sample(sent_ls1[target_word],min(len(sent_ls1[target_word]), len(sent_ls2[target_word])))
        sent_ls2[target_word] = random.sample(sent_ls2[target_word],min(len(sent_ls1[target_word]), len(sent_ls2[target_word])))

    progress_bar = tqdm(range(len(target_list)))
    for target_word in target_list:
        for sent in sent_ls1[target_word]:
            sent_ = sent.split()
            sense = lesk(sent_, target_word)
            if sense not in target_dict1[target_word]:
                target_dict1[target_word][sense] = 1
            else: target_dict1[target_word][sense] += 1
        progress_bar.update(1)


    progress_bar = tqdm(range(len(target_list)))
    for target_word in target_list:
        for sent in sent_ls2[target_word]:
            sent_ = sent.split()
            sense = lesk(sent_, target_word)
            if sense not in target_dict2[target_word]:
                target_dict2[target_word][sense] = 1
            else: target_dict2[target_word][sense] += 1
        progress_bar.update(1)


    dis_dict = {}


    for target_word in target_list:
        sense_set = set(target_dict1[target_word]).union(set(target_dict2[target_word]))
        sense_ids1 = []
        sense_ids2 = []
        for sense in sense_set:
            if sense in target_dict1[target_word]:
                sense_ids1.append(target_dict1[target_word][sense])
            else: sense_ids1.append(0.0)
            if sense in target_dict2[target_word]:
                sense_ids2.append(target_dict2[target_word][sense])
            else: sense_ids2.append(0.0)
        # sum1 = sum(sense_ids1)
        # sum2 = sum(sense_ids2)
        # sense_ids1 = [x/sum1 for x in sense_ids1]
        # sense_ids2 = [x/sum2 for x in sense_ids2]

        # if 0 in sense_ids1: sense_ids1 = [float(x + 0.00001) for x in sense_ids1]
        # if 0 in sense_ids2: sense_ids2 = [float(x + 0.00001) for x in sense_ids2]
        
        # dis_dict[target_word] = sum(kl_div(sense_ids1,sense_ids2))
        dis_dict[target_word] = jensenshannon(sense_ids1,sense_ids2)


    # for target_word in target_list:
    #     jaccard_dict[target_word] = new_score(target_dict1[target_word], target_dict2[target_word])

    with open(f'{args.results_dir}/english/hypothesis/senseset_c1.txt', 'w') as f:
        for target_word in target_list:
            f.write(f'{target_word}')
            for sense in target_dict1[target_word]:
                f.write(f',{sense}:{target_dict1[target_word][sense]}')
            f.write('\n')
    with open(f'{args.results_dir}/english/hypothesis/senseset_c2.txt', 'w') as f:
        for target_word in target_list:
            f.write(f'{target_word}')
            for sense in target_dict2[target_word]:
                f.write(f',{sense}:{target_dict2[target_word][sense]}')
            f.write('\n')
    with open(f'{args.results_dir}/english/hypothesis/dis_dict.txt', 'w') as f:
        for target_word in target_list:
            f.write(f'{target_word},{dis_dict[target_word]}\n')
    
    with open(f'{args.results_dir}/english/hypothesis/score.txt', 'w') as f:
        for target_word in target_list:
            f.write(f'{dis_dict[target_word]},')

    # with open(f'{args.results_dir}/english/hypothesis/new_score.txt', 'w') as f:
    #     for target_word in target_list:
    #         f.write(f'{target_word},{new_dict[target_word]}\n')

    truth = []
    with open('data/english/truth/graded.txt') as f:
        lines = f.readlines()
        for l in lines:
            truth.append(l.split()[-1])

    new = [dis_dict[target_word] for target_word in target_list]
    score = stats.spearmanr(truth,new)[0]
    print(score)

if __name__ == '__main__':

    start_time = time.time()

    main()

    print('---------- {:.1f} minutes ----------'.format((time.time() - start_time) / 60))
    print()