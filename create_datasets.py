import argparse
from glob import glob
import gzip
import logging
import os

from conllu import parse_incr
import tqdm


logging.basicConfig(level=logging.INFO)
LANGUAGES = (
    'german',
    'norwegian1',
    'norwegian2',
    'russian',
    'english',
    'latin',
    'italian',
    'swedish',
)
PROMPTS = {
    "english": "What is the definition of"
}
POS = {"NOUN": "nn", "VERB": "vb"}
LEMMA = "lemma"


def parse_arge():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--lang",
        default="english",
        choices=LANGUAGES,
    )
    parser.add_argument("--conll_path", default="~/corpora-acl/parsed")
    parser.add_argument(
        "--targets_path",
        default="~/Downloads/semeval2020_ulscd_eng/targets.txt",
    )
    parser.add_argument(
        "--res_path",
        default="../acl_data"
    )
    parser.add_argument(
        "--n_first",
        default=None,
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arge()
    lang_path = os.path.join(os.path.expanduser(args.conll_path), args.lang)
    with open(os.path.expanduser(args.targets_path), "r") as targets_file:
        targets = targets_file.readlines()
    targets = [target.strip() for target in targets]

    for corpus in glob(f"{lang_path}/*.gz"):
        logging.info(corpus)
        prompts, targets_list = [], []
        with gzip.open(corpus, "rt") as corpus_file:
            count = 0
            for token_list in tqdm.tqdm(parse_incr(corpus_file)):
                if (args.n_first is None) or (count < args.n_first):
                    if args.lang == "english":
                        lemmas_set = {
                            f'{tok[LEMMA]}_{POS.get(tok["upos"])}' for tok
                            in token_list
                        }
                    else:
                        lemmas_set = {tok[LEMMA] for tok in token_list}

                    intersected = lemmas_set.intersection(targets)
                    if intersected:
                        sent = ' '.join([tok["form"] for tok in token_list])
                        for target in intersected:
                            target = target.split("_")[0]
                            prompts.append(
                                f"{sent} {PROMPTS[args.lang]} {target}?",
                            )
                            targets_list.append(target)
                            count += 1
                else:
                    break
        res_path = os.path.join(
            os.path.expanduser(args.res_path),
            f"{corpus.replace('/', '-')}{os.extsep}txt{os.extsep}gz",
        )
        with gzip.open(res_path, "wt") as results_file:
            for target, prompt in zip(targets_list, prompts):
                results_file.write(f"{target}\t{prompt}\n")
