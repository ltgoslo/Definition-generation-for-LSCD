import argparse
from collections import Counter
import logging
import os
import random
import time

import pandas as pd
from scipy import stats
from scipy.spatial.distance import (
    jensenshannon,
    braycurtis,
    canberra,
    chebyshev,
    cosine,
    euclidean,
)
import numpy as np

from utils import kl

METRICS_NAMES = [
    "Cosine",
    "Chebyshev",
    "Canberra",
    "Bray-Curtis",
    "Euclidean",
    "JS",
    "KL",
]

METRICS = [
    cosine,
    chebyshev,
    canberra,
    braycurtis,
    euclidean,
    jensenshannon,
    kl,
]
method = "defgen"


def get_senses_defgen(target_dict, data_dir, language, period):
    filename = f"{language}-corpus{period}.tsv.gz"
    datafile = os.path.join(data_dir, filename)
    corpus = pd.read_csv(datafile, sep="\t", header=None)
    corpus.columns = ["word", "usage", "definition"]
    for target_word in corpus.word.unique():
        this_word = corpus[corpus.word == target_word]
        target_dict[target_word] = Counter(this_word.definition)
        logging.debug(
            f"{target_word}: {len(target_dict[target_word])} unique senses")
    return target_dict


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        default="../../acl_data",
        type=str,
        help="Data directory with gold data",
    )
    parser.add_argument(
        "--results_dir",
        default="../../acl_results",
        type=str,
        help="Results directory.",
    )
    parser.add_argument(
        "--defgen_path",
        help="Path to the directory with generated definitions",
        default="../../acl_results/generated_definitions/merged",
    )
    parser.add_argument("--output")
    return parser.parse_args()


def write_results(
        args,
        target_list,
        target_dict1,
        target_dict2,
        dis_dicts,
        truth,
        target_list_original,
        lang,
):
    logging.info("=========")
    logging.info("Evaluation")
    logging.info("=========")
    if not os.path.exists(args.results_dir):
        os.mkdir(args.results_dir)
    method_dir = os.path.join(args.results_dir, method)
    if not os.path.exists(method_dir):
        os.mkdir(os.path.join(method_dir))
    method_dir = os.path.join(method_dir, lang)
    if not os.path.exists(method_dir):
        os.mkdir(method_dir)
    with open(f"{method_dir}/senseset_c1.txt", "w") as f:
        for target_word in target_list:
            f.write(f"{target_word}")
            for sense in target_dict1[target_word]:
                f.write(f",{sense}:{target_dict1[target_word][sense]}")
            f.write("\n")
    with open(f"{method_dir}/senseset_c2.txt", "w") as f:
        for target_word in target_list:
            f.write(f"{target_word}")
            for sense in target_dict2[target_word]:
                f.write(f",{sense}:{target_dict2[target_word][sense]}")
            f.write("\n")

    for metric, dis_dict in zip(METRICS_NAMES, dis_dicts):
        with open(f"{method_dir}/{metric}_dict.tsv", "w") as f:
            for i, target_word in enumerate(target_list_original):
                f.write(
                    f"{target_word}\t{dis_dict[target_list[i]]}\n")

        with open(f"{method_dir}/{metric}_score.txt", "w") as f:
            for target_word in target_list:
                f.write(f"{dis_dict[target_word]},")

        new = [dis_dict[target_word] for target_word in
               target_list]
        score, p = stats.spearmanr(truth, new, nan_policy='omit')
        logging.info(f"{method}, {metric}: {round(score, 3)}, p-value {p}")


def main():
    random.seed(123)
    np.random.seed(123)
    args = parse_args()
    logging.basicConfig(
        format="%(asctime)s : %(levelname)s : %(message)s",
        level=logging.INFO,
        filename=f"logs/{args.output}.log"
    )
    for lang in (
            "english",
            "russian1",
            "russian2",
            "russian3",
            "norwegian1",
            "norwegian2"
    ):
        logging.info(lang.upper())
        for merg_strat in ("maximal", "minimal"):
            logging.info(merg_strat)
            for strat in ("beam", "divbeam", "greedy", "greedy10", "beam50", "divbeam50", "greedy50"):
                logging.info(strat)
                target_dict1 = {}
                target_dict2 = {}
                sent_ls1 = {}
                sent_ls2 = {}
                target_list, target_list_original = [], []
                truth, target_list_original = [], []
                with open(f"{args.data_dir}/{lang}/truth/graded.txt") as f:
                    lines = f.readlines()
                    for el in lines:
                        splitted = el.split()
                        target_list_original.append(splitted[0])
                        truth.append(float(splitted[-1]))
                with open(os.path.join(args.data_dir,
                                       f"{lang}/targets.txt")) as target_file:
                    for line in target_file:
                        if (line.rstrip() in target_list_original) or (
                                line.rstrip() in {"formiddag",
                                                  "land"}):  # not all Norwegian words are used
                            target_word = line.strip().split("_")[0]
                            target_list.append(target_word)
                            target_dict1[target_word] = {}
                            target_dict2[target_word] = {}
                            sent_ls1[target_word] = []
                            sent_ls2[target_word] = []

                    defgen_path = os.path.join(args.defgen_path, lang, merg_strat, strat)
                    try:
                        target_dict1 = get_senses_defgen(target_dict1, defgen_path,
                                                         lang, "1")
                    except FileNotFoundError:
                        continue
                    target_dict2 = get_senses_defgen(target_dict2, defgen_path,
                                                     lang, "2")
                    dis_dicts = [{} for _ in METRICS]
                    for target_word in target_list:
                        sense_set = set(target_dict1[target_word]).union(
                            set(target_dict2[target_word]))
                        sense_ids1 = []
                        sense_ids2 = []
                        for sense in sense_set:
                            if sense in target_dict1[target_word]:
                                sense_ids1.append(target_dict1[target_word][sense])
                            else:
                                sense_ids1.append(0.0)
                            if sense in target_dict2[target_word]:
                                sense_ids2.append(target_dict2[target_word][sense])
                            else:
                                sense_ids2.append(0.0)
                        for i, metric in enumerate(METRICS[:-1]):
                            if sense_ids1:
                                dis_dicts[i][target_word] = metric(sense_ids1,
                                                                   sense_ids2)
                            else:
                                dis_dicts[i][target_word] = 0.00001
                        if sense_ids1:
                            dis_dicts[-1][target_word] = METRICS[-1](
                                sense_ids1,
                                sense_ids2,
                                True,
                            )
                        else:
                            dis_dicts[-1][target_word] = 0.00001
                    write_results(
                        args,
                        target_list,
                        target_dict1,
                        target_dict2,
                        dis_dicts,
                        truth,
                        target_list_original,
                        lang,
                    )
                    logging.info(
                        "Don't forget to evaluate the predictions with the official scorer.")
                    logging.info(
                        f"They can be found in the *.dict.tsv files in the "
                        f"{os.path.join(args.results_dir, method, lang)} directory.")


if __name__ == "__main__":
    start_time = time.time()
    main()
    logging.info(
        "---------- {:.1f} minutes ----------".format(
            (time.time() - start_time) / 60)
    )
