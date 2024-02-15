#! /bin/env python3
# coding: utf-8

import argparse
import logging
import os
from collections import Counter
from os import path
from unicodedata import category
import pandas as pd
from leven import levenshtein


def normalize(text, badwords):
    text = "".join(ch for ch in text if category(ch)[0] != "P")
    if badwords:
        tokenized = text.split()
        text = [w for w in tokenized if not w.lower() in badwords]
        text = " ".join(text)
    text = text.strip()
    return text


def find_merges(df, argums):
    targ_words = df.word.unique()

    if argums.strategy == "minimal":
        logging.info("Minimalist: finding the definitions similar to the most frequent one...")
    else:
        logging.info("Full merging: finding the definitions similar to each other...")

    # Finding definitions similar to the most frequent one (paraphrases) for each targ_word:
    mappings = {targ_word: {} for targ_word in targ_words}

    for targ_word in targ_words:
        definitions = Counter(df[df.word == targ_word].definition).most_common()
        # definitions = sorted(definitions)  # can be commented to out to start from most frequent
        logging.debug(f"{targ_word}: {len(definitions)} unique senses before merging")
        sim_cache = {}
        def2compare = None
        for nr, source in enumerate(definitions):
            cand = source[0]
            if len(cand.split()) >= LENGTH and cand not in mappings[targ_word]:
                def2compare = cand
                mapped = 0
                for definition in definitions:
                    definition_text = definition[0]
                    if definition_text != def2compare and \
                            definition_text not in mappings[targ_word]:
                        if (definition_text, def2compare) in sim_cache:
                            distance = sim_cache[(definition_text, def2compare)]
                        else:
                            distance = levenshtein(definition_text, def2compare)
                            sim_cache[(definition_text, def2compare)] = distance
                            sim_cache[(def2compare, definition_text)] = distance
                        if distance < DISTANCE:
                            mappings[targ_word][definition_text] = def2compare
                            mapped += 1
                if mapped > 0:
                    logging.debug(f"{mapped} definitions mapped to {def2compare}")
            if argums.strategy == "minimal":
                if def2compare:
                    break
        if not def2compare:
            logging.info(f"No dominant sense with a long enough definition found for {targ_word}!")
    logging.info("==================================")
    return mappings, targ_words


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path",
        type=str,
        help="Directory with two time-specific datasets",
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
        "--thresh",
        type=int,
        help="Levenshtein distance threshold: we merge only definitions longer than thresh words",
        default=50
    )
    parser.add_argument(
        "--len",
        type=int,
        help="Minimal length of the definition (in words) to be allowed "
             "to replace other definitions",
        default=4,
    )
    parser.add_argument(
        "--out",
        type=str,
        help="Directory to save datasets with merged definitions",
        required=True
    )
    parser.add_argument(
        "--strategy",
        choices=["maximal", "minimal"],
        help="Use only the most frequent definition to merge with (minimal) "
             "or all definitions (maximal)",
        default="maximal"
    )
    args = parser.parse_args()

    os.makedirs(path.dirname(args.out), exist_ok=True)

    LENGTH = args.len

    DISTANCE = args.thresh

    logging.basicConfig(format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO)

    filename1 = f"{args.lang}-corpus1.tsv.gz"
    filename2 = f"{args.lang}-corpus2.tsv.gz"
    df1 = pd.read_csv(path.join(args.data_path, filename1), sep="\t", header=None)
    df2 = pd.read_csv(path.join(args.data_path, filename2), sep="\t", header=None)
    df1.columns = ["word", "usage", "definition"]
    df2.columns = ["word", "usage", "definition"]

    # It seems stopwords do not really help
    # notcontent = set(stopwords.words("norwegian" if "norwegian" in args.lang else args.lang))

    notcontent = None
    if args.punct:
        logging.info("Removing punctuation marks from the definitions...")
        df1["definition"] = df1["definition"].apply(lambda x: normalize(x, notcontent))
        df2["definition"] = df2["definition"].apply(lambda x: normalize(x, notcontent))

    # df = pd.concat([df1, df2])  # Concatenated dataset with definitions from both periods

    for period, filename in zip([df1, df2], [filename1, filename2]):
        logging.info(f"Processing {filename}")
        mapping, words = find_merges(period, args)
        for word in words:
            new_defs = [
                mapping[word][d] if d in mapping[word] else d
                for d in period[period.word == word].definition
            ]
            period.loc[period.word == word, "definition"] = new_defs
            cur_definitions = Counter(period[period.word == word].definition).most_common()
            logging.debug(f"{word}: {len(cur_definitions)} unique senses")
        period.to_csv(path.join(args.out, filename), sep="\t", header=False, index=False)
        logging.info(f"Merged definitions saved to {path.join(args.out, filename)}")
