import argparse
from glob import glob
import os
import re
import csv

import pandas as pd

PATTERN_NB = re.compile(r"Hva betyr (\w+)\?")
PATTERNS = {
    'english': re.compile(r"What is the definition of (\w+)\?"),
    'russian': re.compile(r"Что такое (\w+)\?"),
    'norwegian1': PATTERN_NB,
    'norwegian2': PATTERN_NB,
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir", default='../generated_definitions/russian/greedy/',
    )
    parser.add_argument(
        "--results_dir",
        default='generated_definitions',
    )
    parser.add_argument('--rm_whole_prompt', default=2, type=int, choices=(0, 1, 2))
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    splitted = args.data_dir.split('/')
    language = splitted[-3]
    gen = splitted[-2]
    res_path = os.path.join(args.results_dir, language, gen)
    if not os.path.exists(res_path):
        os.makedirs(res_path)
    quoting = csv.QUOTE_MINIMAL
    if language == 'english':
        quoting = csv.QUOTE_NONE
    for corpus in glob(f"{args.data_dir}/*.tsv.gz"):
        data = pd.read_csv(corpus, sep='\t', compression='gzip', header=None)
        out = os.path.join(res_path, os.path.split(corpus)[-1])
        if args.rm_whole_prompt == 0:
            data.drop(data.columns[1], axis=1, inplace=True)
        elif args.rm_whole_prompt == 1:  # remove What is the definition...? only
            data.iloc[:,1] = data.iloc[:,1].apply(
                lambda x: re.sub(PATTERNS[language], '', x)
            )
        else: # remove usage example
            data.iloc[:,1] = data.iloc[:,1].apply(
                lambda x: re.search(PATTERNS[language], x).group(0)
            )
        data.to_csv(
            out,
            sep='\t',
            compression='gzip',
            quoting=quoting,
            index=False,
            header=False,
        )
