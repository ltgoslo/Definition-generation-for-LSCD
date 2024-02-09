import argparse
from statistics import median
import logging
import os

from ax import optimize
from ax.exceptions.core import UserInputError
import random
import numpy as np
import math
import ruptures as rpt
import torch
from sklearn.metrics import accuracy_score

from sense_dis import METRICS_NAMES

RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
logging.basicConfig(level=logging.INFO)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--sense_dis_method_dir',
        default=None,
        type=str,
        required=True,
        help='sense_dis.py results directory.',
    )
    parser.add_argument(
        "--method", default="median", choices=(
            "ax", "median", "change_point",
        ),
    )
    return parser.parse_args()


def detect_change_point(sequence, n_chp=1):
    """
    Detects the indices of change points in a sequence of values
    """
    sequence = np.array(sequence)
    algo = rpt.Dynp(model="rbf", jump=1).fit(sequence)
    chp_index, length = algo.predict(n_bkps=n_chp)
    return chp_index


def cal_acc(p):
    y_pred = []
    for i in dev_score:
        if i >= p['threshold']:
            y_pred.append(1)
        else:
            y_pred.append(0)
    return accuracy_score(y_pred, y_dev_true)


Y_TRUE = (
    1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0,
    1,
    1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0)  # EN #TODO: read from file
y_true_len = len(Y_TRUE)
y_dev_index = random.sample(range(0, y_true_len),
                            k=int(math.ceil(0.2 * y_true_len)))
y_dev_true = [Y_TRUE[x] for x in y_dev_index]

args = parse_args()
for metric in METRICS_NAMES:
    logging.info(metric)
    with open(
            os.path.join(args.sense_dis_method_dir, f"{metric}_score.txt"),
            "r",
            encoding="utf8",
    ) as scores_file:
        score = [float(x) for x in scores_file.read().rstrip().split(",") if x]

    if args.method == "ax":
        dev_score = [score[x] for x in y_dev_index]
        logging.info(f"length of truth: {y_true_len}")
        logging.info(f"length of scores: {len(score)}")
        assert len(score) == y_true_len
        # word_ls = ['attack','bag','ball','bit','chairman','circle','contemplation','donkey','edge','face','fiction','gas','graft','head','land','lane','lass','multitude','ounce','part','pin','plane','player','prop','quilt','rag','record','relationship','risk','savage','stab','stroke','thump','tip','tree','twist','word']

        bound_score = [x for x in score if
                       x != float('inf') and x != float('-inf')]
        for x in score:
            if x == float('inf'): bound_score.append(2 * max(bound_score))
            if x == float('-inf'): bound_score.append(
                min(bound_score) - abs(min(bound_score)))
        logging.info(bound_score)
        try:
            best_parameters, best_values, _, _ = optimize(
                parameters=[
                    {"name": "threshold",
                     "type": "range",
                     "bounds": [float(min(bound_score)),
                                float(max(bound_score))], }, ],
                evaluation_function=cal_acc,
                minimize=False, random_seed=RANDOM_SEED,
            )
        except UserInputError:
            logging.info(f"impossible to optimize for the metric {metric}")
            continue
        logging.info(f"Best parameters: {best_parameters}")
        logging.info(f'best_acc: {cal_acc(best_parameters)}')
        threshold = best_parameters['threshold']
    elif args.method == "median":
        threshold = median(score)
    elif args.method == "change_point":
        threshold = detect_change_point(score)

    y_pred = [1 if x > threshold else 0 for x in score]
    con = []
    for i in range(len(y_pred)):
        if y_pred[i] == 1:
            if Y_TRUE[i]:
                con.append('TP')
            else:
                con.append('FP')
        else:
            if not Y_TRUE[i]:
                con.append('TN')
            else:
                con.append('FN')

    logging.info(round(accuracy_score(y_pred, Y_TRUE), 3))
    logging.info(y_pred)
    logging.info(con)
