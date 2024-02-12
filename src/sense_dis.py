import argparse
from collections import Counter
from copy import copy
import logging
import os
import random
import time
import json
import re
import pandas as pd
import nltk
from scipy import stats
from scipy.spatial.distance import (
    jensenshannon,
    braycurtis,
    canberra,
    chebyshev,
    cosine,
    euclidean,
)
from scipy.special import kl_div
nltk.download("wordnet")
from nltk.wsd import lesk
from tqdm import tqdm
import numpy as np

METRICS_NAMES = [
    "Cosine",
    "Chebyshev",
    "Canberra",
    "Bray-Curtis",
    "Euclidean",
    "JS",
    "KL",
]

logging.basicConfig(format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO)


def kl(sense_ids1, sense_ids2, no_zeros=False):
    sum1 = sum(sense_ids1)
    sum2 = sum(sense_ids2)
    sense_ids1 = [x / sum1 for x in sense_ids1]
    sense_ids2 = [x / sum2 for x in sense_ids2]

    if no_zeros:
        if 0 in sense_ids1:
            sense_ids1 = [float(x + 0.00001) for x in sense_ids1]
        if 0 in sense_ids2:
            sense_ids2 = [float(x + 0.00001) for x in sense_ids2]
    return sum(kl_div(sense_ids1, sense_ids2))


METRICS = [
    cosine,
    chebyshev,
    canberra,
    braycurtis,
    euclidean,
    jensenshannon,
    kl,
]


def _get_senses_lesk(args, target_dict, target_list, target_list_pos, sent_ls):
    for i, target_word in enumerate(tqdm(target_list)):
        word_without_pos = None
        pos = None
        if args.use_pos_in_lesk:
            word_without_pos, pos = target_list_pos[i].split("_")
        for sent in sent_ls[target_word]:
            sent_ = sent.split()
            if args.use_pos_in_lesk:
                # pos tag in SemEval shared task lemma corpus are nn, vb for English
                # and they are n, v in WordNet
                sense = lesk(sent_, word_without_pos, pos[0])
            else:
                sense = lesk(sent_, target_word)
            if sense not in target_dict[target_word]:
                target_dict[target_word][sense] = 1
            else:
                target_dict[target_word][sense] += 1
    return target_dict


def get_senses_lesk(
    args,
    target_list,
    target_list_pos,
    sent_ls1,
    sent_ls2,
    target_dict1,
    target_dict2,
):
    # should be those that are lemmatized as I understand
    with open(os.path.join(args.data_dir, "english/corpus1/ccoha1.txt"), "r") as ccoha1:
        c1_text = [line.strip() for line in ccoha1.readlines()]

    with open(os.path.join(args.data_dir, "english/corpus2/ccoha2.txt"), "r") as ccoha2:
        c2_text = [line.strip() for line in ccoha2.readlines()]

    logging.info(f"{len(c1_text)}, {len(c2_text)}")

    for i, target_word in enumerate(tqdm(target_list)):
        word = copy(target_word)
        for sent in c1_text:
            sent = re.sub(r"_\w+", "", sent)
            if word in sent.split():
                sent_ls1[target_word].append(sent)
        for sent in c2_text:
            sent = re.sub(r"_\w+", "", sent)
            if word in sent.split():
                sent_ls2[target_word].append(sent)

        logging.info(
            f"{target_word}, {len(sent_ls1[target_word])}, {len(sent_ls2[target_word])}"
        )
        sent_ls1[target_word] = random.sample(
            sent_ls1[target_word],
            min(len(sent_ls1[target_word]), len(sent_ls2[target_word])),
        )
        sent_ls2[target_word] = random.sample(
            sent_ls2[target_word],
            min(len(sent_ls1[target_word]), len(sent_ls2[target_word])),
        )

    target_dict1 = _get_senses_lesk(
        args, target_dict1, target_list, target_list_pos, sent_ls1
    )
    with open(
        f"{args.results_dir}/english/hypothesis/sent_ls1.json", "w", encoding="utf8"
    ) as f:
        json.dump(sent_ls1, f)
    target_dict2 = _get_senses_lesk(
        args, target_dict2, target_list, target_list_pos, sent_ls2
    )
    with open(
        f"{args.results_dir}/english/hypothesis/sent_ls2.json", "w", encoding="utf8"
    ) as f:
        json.dump(sent_ls2, f)
    return target_dict1, target_dict2


def get_senses_defgen(target_dict, data_dir, language, period):
    filename = f"{language}-corpus{period}.tsv.gz"
    datafile = os.path.join(data_dir, filename)
    corpus = pd.read_csv(datafile, sep="\t", header=None)
    corpus.columns = ["word", "usage", "definition"]
    for target_word in corpus.word.unique():
        this_word = corpus[corpus.word == target_word]
        target_dict[target_word] = Counter(this_word.definition)
        logging.debug(f"{target_word}: {len(target_dict[target_word])} unique senses")
    return target_dict


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        default=None,
        type=str,
        required=True,
        help="Data directory with gold data and lemmatized corpus for Lesk",
    )
    parser.add_argument(
        "--results_dir",
        default=None,
        type=str,
        required=True,
        help="Results directory.",
    )
    parser.add_argument("--method", choices=["lesk", "defgen"], default="lesk")
    parser.add_argument("--use_pos_in_lesk", type=bool, default=False)
    parser.add_argument("--no_zeros_in_kl", type=bool, help="Use smoothing in KL distance",
                        default=True)
    parser.add_argument("--lang", default="english")
    parser.add_argument(
        "--defgen_path",
        help="Path to the directory with generated definitions",
        default="/cluster/projects/nn9851k/andreku/defgen_lscd/generated_definitions/english/diverse_beam_search",
    )
    return parser.parse_args()


def write_results(args, target_list, target_dict1, target_dict2, dis_dicts):
    logging.info("=========")
    logging.info("Evaluation")
    logging.info("=========")
    if not os.path.exists(args.results_dir):
        os.mkdir(args.results_dir)
    method_dir = os.path.join(args.results_dir, args.method, args.lang)
    if not os.path.exists(method_dir):
        os.mkdir(os.path.join(args.results_dir, args.method))
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
            for target_word in target_list:
                f.write(f"{target_word}\t{dis_dict[target_word]}\n")

        with open(f"{method_dir}/{metric}_score.txt", "w") as f:
            for target_word in target_list:
                f.write(f"{dis_dict[target_word]},")

        truth = []
        with open(f"{args.data_dir}/{args.lang}/truth/graded.txt") as f:
            lines = f.readlines()
            for el in lines:
                truth.append(el.split()[-1])

        new = [dis_dict[target_word] for target_word in target_list]
        score = stats.spearmanr(truth, new)[0]
        logging.info(f"{args.method}, {metric}: {round(score, 3)}")


def main():
    random.seed(123)
    np.random.seed(123)
    args = parse_args()

    target_dict1 = {}
    target_dict2 = {}
    sent_ls1 = {}
    sent_ls2 = {}
    target_list, target_list_pos = [], []

    with open(os.path.join(args.data_dir, f"{args.lang}/targets.txt")) as target_file:
        for line in target_file:
            target_list_pos.append(line.rstrip())
            target_word = line.split("_")[0]
            target_list.append(target_word)
            target_dict1[target_word] = {}
            target_dict2[target_word] = {}
            sent_ls1[target_word] = []
            sent_ls2[target_word] = []
    if args.method == "lesk":
        target_dict1, target_dict2 = get_senses_lesk(
            args,
            target_list,
            target_list_pos,
            sent_ls1,
            sent_ls2,
            target_dict1,
            target_dict2,
        )
    elif args.method == "defgen":
        target_dict1 = get_senses_defgen(target_dict1, args.defgen_path, args.lang, "1")
        target_dict2 = get_senses_defgen(target_dict2, args.defgen_path, args.lang, "2")

    dis_dicts = [{} for _ in METRICS]
    for target_word in target_list:
        sense_set = set(target_dict1[target_word]).union(set(target_dict2[target_word]))
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
        if target_word == "multitude":
            print(f"bit1: {sense_ids1}")
            print(f"bit2: {sense_ids2}")
            print(cosine(sense_ids1, sense_ids2))
        for i, metric in enumerate(METRICS[:-1]):
            dis_dicts[i][target_word] = metric(sense_ids1, sense_ids2)
        dis_dicts[-1][target_word] = METRICS[-1](
            sense_ids1,
            sense_ids2,
            args.no_zeros_in_kl,
        )
    write_results(args, target_list, target_dict1, target_dict2, dis_dicts)
    logging.info("Don't forget to evaluate the predictions with the official scorer.")
    logging.info(f"They can be found in the *.dict.tsv files in the "
                 f"{os.path.join(args.results_dir, args.method, args.lang)} directory.")


if __name__ == "__main__":
    start_time = time.time()
    main()
    logging.info(
        "---------- {:.1f} minutes ----------".format((time.time() - start_time) / 60)
    )

