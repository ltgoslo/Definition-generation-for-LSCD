import argparse
from ast import literal_eval
from collections import Counter
import logging
import os
import random
import time
import json
import re
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
from tqdm import tqdm
import numpy as np

from utils import kl, lesk, read_html_wiktionary

METRICS_NAMES = [
    "Cosine",
    "Chebyshev",
    "Canberra",
    "Bray-Curtis",
    "Euclidean",
    "JS",
    "KL",
]

logging.basicConfig(
    format="%(asctime)s : %(levelname)s : %(message)s",
    level=logging.INFO,
)
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
    dictionary, synsets = None, None
    if "norwegian" in args.lang:
        dictionary = pd.read_csv(
            os.path.expanduser(
                os.path.join(
                    args.data_dir,
                    args.lang,
                    "complete.tsv.gz",
                ),
            ),
            sep="\t",
            compression="gzip",
        )
        # SYM - symbol
        # COMPPFX - first part of a compound
        # PFX - prefix
        # EXPR - expression
        logging.info(f"Unique Norwegian POS tags: {dictionary['POS'].unique()}")
    elif "russian" in args.lang:
        dictionary = pd.read_csv(
            os.path.expanduser(
                os.path.join(
                    args.data_dir,
                    args.lang,
                    "ru.complete.csv",
                ),
            ),
        )
    for i, target_word in enumerate(tqdm(target_list)):
        if ("norwegian" in args.lang) or ("russian" in args.lang):
            synsets = dictionary[
                dictionary["word"] == target_word
            ].drop_duplicates("gloss")
            if args.use_pos_in_lesk and ("norwegian" in args.lang):
                # we know from the NorDiaChange paper that all words are nouns
                synsets = synsets[synsets["POS"].isin({"NOUN", "PROPN"})]
            if synsets.shape[0] == 0:
                if "norwegian" in args.lang:
                    continue
                elif "russian" in args.lang:
                    synsets = read_html_wiktionary(target_word)
                    if synsets.shape[0] == 0:
                        continue
        word_without_pos = None
        pos = None
        if args.use_pos_in_lesk:
            if args.lang == "english":
                word_without_pos, pos = target_list_pos[i].split("_")
            elif ("norwegian" in args.lang) or ("russian" in args.lang):
                word_without_pos, pos = target_list_pos[i], ["NOUN"]
        for sent in sent_ls[target_word]:
            if args.use_pos_in_lesk:
                # pos tag in SemEval shared task lemma corpus are nn, vb for English
                # and they are n, v in WordNet
                sense = lesk(
                    sent,
                    word_without_pos,
                    pos[0],
                    synsets=synsets, lang=args.lang,
                )
            else:
                sense = lesk(sent, target_word, synsets=synsets,
                             lang=args.lang)
            if sense not in target_dict[target_word]:
                target_dict[target_word][sense] = 1
            else:
                target_dict[target_word][sense] += 1
    return target_dict


def load_corpora(args):
    # should be those that are lemmatized
    if args.lang == "english":
        with open(
                os.path.join(args.data_dir, f"{args.lang}/corpus1/ccoha1.txt"),
                "r",
        ) as ccoha1:
            c1_text = [line.strip() for line in ccoha1.readlines()]

        with open(
                os.path.join(args.data_dir, f"{args.lang}/corpus2/ccoha2.txt"),
                "r",
        ) as ccoha2:
            c2_text = [line.strip() for line in ccoha2.readlines()]
    elif ("norwegian" in args.lang) or ("russian" in args.lang):
        c_texts = []
        for period in (1, 2):
            filename = f"{args.lang}/{args.lang}-corpus{period}.tsv.gz"
            datafile = os.path.join(args.data_dir, filename)
            corpus = pd.read_csv(
                datafile,
                sep="\t",
                header=None,
                compression="gzip",
            )
            c_texts.append(corpus[1].to_list())
        c1_text, c2_text = c_texts
    logging.info(f"{len(c1_text)}, {len(c2_text)}")
    return c1_text, c2_text


def get_senses_lesk(
        args,
        target_list,
        target_list_pos,
        sent_ls1,
        sent_ls2,
        target_dict1,
        target_dict2,
):
    lang_folder = f"{args.results_dir}/{args.lang}"
    sent_ls1_path = f"{lang_folder}/sent_ls1.json"
    sent_ls2_path = sent_ls1_path.replace("ls1", "ls2")
    if not os.path.exists(sent_ls1_path):
        if not os.path.exists(lang_folder):
            os.mkdir(lang_folder)
        c1_text, c2_text = load_corpora(args)
        for i, target_word in enumerate(tqdm(target_list)):
            for sent in c1_text:
                sent = re.sub(r"_\w+", "", sent)
                if target_word in sent.split():
                    sent_ls1[target_word].append(sent)
            for sent in c2_text:
                sent = re.sub(r"_\w+", "", sent)
                if target_word in sent.split():
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
        with open(sent_ls1_path, "w", encoding="utf8") as f:
            json.dump(sent_ls1, f)
        with open(sent_ls2_path, "w",
                  encoding="utf8") as f:
            json.dump(sent_ls2, f)
    else:
        with open(sent_ls1_path, "r", encoding="utf8") as f:
            sent_ls1 = json.load(f)
        with open(sent_ls2_path, "r", encoding="utf8") as f:
            sent_ls2 = json.load(f)
    target_dict1 = _get_senses_lesk(
        args, target_dict1, target_list, target_list_pos, sent_ls1
    )
    target_dict2 = _get_senses_lesk(
        args, target_dict2, target_list, target_list_pos, sent_ls2
    )
    return target_dict1, target_dict2


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
    parser.add_argument("--use_pos_in_lesk", type=literal_eval, default=False)
    parser.add_argument("--no_zeros_in_kl", type=bool,
                        help="Use smoothing in KL distance",
                        default=True)
    parser.add_argument("--lang", default="english")
    parser.add_argument(
        "--defgen_path",
        help="Path to the directory with generated definitions",
        default="generated_definitions/english/diverse_beam_search",
    )
    return parser.parse_args()


def write_results(
        args, target_list, target_dict1, target_dict2, dis_dicts, truth, target_list_original,
):
    logging.info("=========")
    logging.info("Evaluation")
    logging.info("=========")
    if not os.path.exists(args.results_dir):
        os.makedirs(args.results_dir, exist_ok=True)
    method_dir = os.path.join(args.results_dir, args.method)
    if not os.path.exists(method_dir):
        os.makedirs(os.path.join(method_dir), exist_ok=True)
    method_dir = os.path.join(method_dir, args.lang)
    if args.method == "lesk":
        method_dir += f"_pos_{args.use_pos_in_lesk}"
    if not os.path.exists(method_dir):
        os.makedirs(method_dir, exist_ok=True)
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
        score = stats.spearmanr(truth, new, nan_policy='omit').statistic
        logging.info(f"{args.method}, {metric}: {round(score, 3)}")


def main():
    random.seed(123)
    np.random.seed(123)
    args = parse_args()

    target_dict1 = {}
    target_dict2 = {}
    sent_ls1 = {}
    sent_ls2 = {}
    target_list, target_list_original = [], []
    truth, target_list_original = [], []
    with open(f"{args.data_dir}/{args.lang}/truth/graded.txt") as f:
        lines = f.readlines()
        for el in lines:
            splitted = el.split()
            target_list_original.append(splitted[0])
            truth.append(float(splitted[-1]))
    with open(os.path.join(args.data_dir,
                           f"{args.lang}/targets.txt")) as target_file:
        for line in target_file:
            if (line.rstrip() in target_list_original) or (line.rstrip() in {"formiddag", "land"}):  # not all Norwegian words are used
                target_word = line.strip().split("_")[0]
                target_list.append(target_word)
                target_dict1[target_word] = {}
                target_dict2[target_word] = {}
                sent_ls1[target_word] = []
                sent_ls2[target_word] = []
    if args.method == "lesk":
        target_dict1, target_dict2 = get_senses_lesk(
            args,
            target_list,
            target_list_original,
            sent_ls1,
            sent_ls2,
            target_dict1,
            target_dict2,
        )
    elif args.method == "defgen":
        target_dict1 = get_senses_defgen(target_dict1, args.defgen_path,
                                         args.lang, "1")
        target_dict2 = get_senses_defgen(target_dict2, args.defgen_path,
                                         args.lang, "2")
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
                # Rounding because of numerical precision issues with JSD
                dis_dicts[i][target_word] = round(metric(sense_ids1, sense_ids2), 10)
            else:
                dis_dicts[i][target_word] = 0.00001
        if sense_ids1:
            dis_dicts[-1][target_word] = METRICS[-1](
                sense_ids1,
                sense_ids2,
                args.no_zeros_in_kl,
            )
        else:
            dis_dicts[-1][target_word] = 0.00001
    write_results(
        args, target_list, target_dict1, target_dict2, dis_dicts, truth, target_list_original
    )
    logging.info(
        "Don't forget to evaluate the predictions with the official scorer.")
    logging.info(f"They can be found in the *.dict.tsv files in the "
                 f"{os.path.join(args.results_dir, args.method, args.lang)} directory.")


if __name__ == "__main__":
    start_time = time.time()
    main()
    logging.info(
        "---------- {:.1f} minutes ----------".format(
            (time.time() - start_time) / 60)
    )
