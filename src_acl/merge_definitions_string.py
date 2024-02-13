#! /bin/env python3
# coding: utf-8

from leven import levenshtein
import pandas as pd
from collections import Counter
import argparse
import logging
import os
from unicodedata import category


def normalize(text):
    text = "".join(ch for ch in text if category(ch)[0] != "P")
    text = " ".join(text.split())
    return text


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
        "--punct",
        type=bool,
        help="Remove punctuation marks?",
        default=1
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
    DISTANCE = 10  # We merge only definitions with Levenshtein distance less than that

    logging.basicConfig(format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO)

    df = pd.read_csv(args.data_path, sep="\t", header=None)
    df.columns = ["word", "usage", "definition"]

    if args.punct:
        logging.info("Removing punctuation marks from the definitions...")
        df["definition"] = df["definition"].apply(lambda x: normalize(x))

    words = df.word.unique()

    # Finding definitions similar to the most frequent one (paraphrases) for each word:
    mappings = {word: {} for word in words}
    for word in words:
        definitions = Counter(df[df.word == word].definition).most_common()
        logging.info(f"{word}: {len(definitions)} unique senses")
        # TODO: a better logic for choosing the dominant sense
        # TODO: find all similar pairs
        def2compare = None
        for source in definitions:
            if len(source[0].split()) > LENGTH and source[0] not in mappings[word]:
                def2compare = source[0]
                for definition in definitions:
                    definition_text = definition[0]
                    if definition_text != def2compare and definition_text not in mappings[word]:
                        distance = levenshtein(definition_text, def2compare)
                        if distance < DISTANCE:  # Ad-hoc number, just "very similar"
                            mappings[word][definition_text] = def2compare
                            logging.debug(f"{definition_text} mapped to {def2compare}")
        if not def2compare:
            logging.info(f"No dominant sense with a long enough definition found for {word}!")
        # for paraphrase in mappings[word]:
        #    logging.debug(paraphrase)

    logging.info("==================================")
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
