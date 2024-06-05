import argparse
from ast import literal_eval
from glob import glob
import gzip
import os
import re
import csv

import pandas as pd

PATTERN_RU = re.compile(r"Что такое (\w+)\?")
PATTERN_NB = re.compile(r"Hva betyr (\w+)\?")
PATTERNS = {
    'english': re.compile(r"What is the definition of (\w+)\?"),
    'russian1': PATTERN_RU,
    'russian2': PATTERN_RU,
    'russian3': PATTERN_RU,
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir", default='../generated_definitions/russian1/greedy/',
    )
    parser.add_argument(
        "--results_dir",
        default='generated_definitions',
    )
    parser.add_argument('--rm_whole_prompt', default=True, type=literal_eval)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    language = args.data_dir.split('/')[-2]
    res_path = os.path.join(args.results_dir, language)
    if not os.path.exists(res_path):
        os.mkdir(res_path)
    quoting = csv.QUOTE_MINIMAL
    if language == 'english':
        quoting = csv.QUOTE_NONE
    for corpus in glob(f"{args.data_dir}/*.tsv.gz"):
        data = pd.read_csv(corpus, sep='\t', compression='gzip')
        out = os.path.join(args.res_path, os.path.split(corpus)[-1])
        if args.rm_whole_prompt:
            data.drop([1], axis=1, inplace=True)
        else:
            data[1] = data[1].apply(
                lambda x: re.sub(PATTERNS[language], '', x)
            )
        data.to_csv(out, sep='\t', compression='gzip', quoting=quoting)
