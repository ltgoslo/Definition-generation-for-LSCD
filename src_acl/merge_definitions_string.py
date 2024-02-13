#! /bin/env python3
# coding: utf-8

from leven import levenshtein
import pandas as pd
from collections import Counter
import argparse
import logging
import os


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path",
        type=str,
        help="File with a dataset",
        required=True
    )
    parser.add_argument(
        "--lang",
        type=str,
        default="english"
    )
    parser.add_argument(
        "--out",
        type=str,
        help="File to save the dataset with merged definitions",
        required=True
    )
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    LENGTH = 3  # We merge only definitions longer than LENGTH words
    logging.basicConfig(format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO)

    df = pd.read_csv(args.data_path, sep="\t", header=None)
    df.columns = ["word", "usage", "definition"]

    words = df.word.unique()

    # Finding definitions similar to the most frequent one (paraphrases) for each word:
    mappings = {word: {} for word in words}
    for word in words:
        definitions = Counter(df[df.word == word].definition).most_common()
        logging.info(f"{word}: {len(definitions)} unique senses")
        # TODO: a better logic for choosing the dominant sense
        # TODO: find all similar pairs
        source = None
        for cand in definitions:
            if len(cand[0].split()) > LENGTH:
                source = cand[0]
                break
        if not source:
            logging.info(f"No dominant sense with a long enough definition found for {word}!")
        for definition in definitions:
            definition_text = definition[0]
            if definition_text != source:
                #  and len(definition_text.split()) > LENGTH
                distance = levenshtein(definition_text, source)
                if distance < 10:  # Ad-hoc number, just "very similar"
                    mappings[word][definition_text] = source
        logging.debug(source)
        logging.debug("=====")
        for paraphrase in mappings[word]:
            logging.debug(paraphrase)

    for word in words:
        new_defs = [
            mappings[word][d] if d in mappings[word] else d
            for d in df[df.word == word].definition
        ]
        df.loc[df.word == word, "definition"] = new_defs
        definitions = Counter(df[df.word == word].definition).most_common()
        logging.info(f"{word}: {len(definitions)} unique senses")
    df.to_csv(args.out, sep="\t", header=False, index=False)
    logging.info(f"Merged definitions saved to {args.out}")
