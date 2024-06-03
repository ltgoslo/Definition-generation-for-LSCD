import numpy as np
from docopt import docopt
from scipy.stats import spearmanr
import os


def get_ys(model_answers, true_answers):
    """
    :param model_answers: path to tab-separated answer file (lemma + "\t" + score)
    :param true_answers: path to tab-separated gold answer file (lemma + "\t" + score)
    :return: a numpy array for the model scores, and one for the true scores
    """
    y_hat_tmp = {}
    errors = 0
    with open(model_answers, 'r', encoding='utf-8') as f_in:
        for line in f_in:
            elements = line.strip().split('\t')
            lemma, score = elements[:2]
            if '_' in lemma:
                lemma = lemma.split('_')[0]
            if score == 'nan':
                errors += 1
            y_hat_tmp[lemma] = score
    if errors:
        print('Found %d NaN predictions' % errors)
    y_hat, y = [], []
    with open(true_answers, 'r', encoding='utf-8') as f_in:
        for line in f_in:
            elements = line.strip().split('\t')
            lemma, score = elements[:2]
            if '_' in lemma:
                lemma = lemma.split('_')[0]
            if lemma == 'landet':
                lemma = 'land'
            if lemma == 'formiddagen':
                lemma = 'formiddag'
            y.append(float(score))
            y_hat.append(float(y_hat_tmp[lemma]))

    return np.array(y_hat), np.array(y)


def eval_task1(model_answers, true_answers):
    """
    Computes the Accuracy against the true binary labels as annotated by humans.
    :param model_answers: path to tab-separated answer file (lemma + "\t" + score)
    :param true_answers: path to tab-separated gold answer file (lemma + "\t" + score)
    :return: binary classification accuracy
    """
    y_hat, y = get_ys(model_answers, true_answers)
    accuracy = np.sum(np.equal(y_hat, y)) / len(y)

    return accuracy


def eval_task2(model_answers, true_answers):
    """
    Computes the Spearman's correlation coefficient against the true rank as annotated by humans
    :param model_answers: path to tab-separated answer file (lemma + "\t" + score)
    :param true_answers: path to tab-separated gold answer file (lemma + "\t" + score)
    :return: (Spearman's correlation coefficient, p-value)
    """
    y_hat, y = get_ys(model_answers, true_answers)
    r, p = spearmanr(y_hat, y, nan_policy='omit')

    return r, p


def main():
    """
    Evaluate lexical semantic change detection results.
    """

    # Get the arguments
    args = docopt("""Evaluate lexical semantic change detection results.

    Usage:
        eval.py <task> <modelAnsPath> <trueAnsPath>

    Arguments:
        <task> = 1 or 2
        <modelAnsPath> = path to tab-separated answer file (lemma + "\t" + score)
        <trueAnsPath> = path to tab-separated gold answer file (lemma + "\t" + score)

    """)

    task = args['<task>']
    modelAnsPath = args['<modelAnsPath>']
    trueAnsPath = args['<trueAnsPath>']

    if int(task) == 1:
        acc = eval_task1(modelAnsPath, trueAnsPath)
        print(f"Task\tScore")
        print(f"1\t{acc:.3f}")
    elif int(task) == 2:
        r, p = eval_task2(modelAnsPath, trueAnsPath)
        print(f"Task\tScore\tp-value")
        print(f"2\t{r:.3f}\t{p:.3f}")
    else:
        print(f"Unknown task number {task}")

if __name__ == '__main__':
    main()
