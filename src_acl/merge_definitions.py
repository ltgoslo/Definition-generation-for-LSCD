import argparse
from copy import deepcopy
from collections import defaultdict
import csv
import logging
import os
import random

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import pairwise_distances
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
WORD_COLUMN = "Word"
DEFS_COLUMN = "Definitions"
SENSE_ID_COLUMN = "sense_id"
TIME_PERIOD_COLUMN = "period"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", default="../../acl_results/embeddings/runB")
    parser.add_argument('--results_dir', default=None, type=str, required=True,
                        help='Results directory.')
    return parser.parse_args()


def load_np(path, period):
    wordtable = pd.read_csv(
        os.path.join(path, f"definitions.english{period}.tsv"),
        sep='\t',
        quoting=csv.QUOTE_NONE,
    )
    embs = np.load(open(os.path.join(path, f"embeddings_perword.english{period}.npz"), "rb"))
    return wordtable, embs


def create_threshold_dict(target_words, threshold_dict, wordtable, period, embs):
    for word in target_words:
        this_word = wordtable[wordtable[WORD_COLUMN] == word]
        defs=this_word[DEFS_COLUMN]
        threshold_dict[DEFS_COLUMN].extend(
            defs.to_list()
        )
        threshold_dict[WORD_COLUMN].extend(
            [word for _ in range(this_word.shape[0])]
        )

        threshold_dict[TIME_PERIOD_COLUMN].extend(
            [period for _ in range(this_word.shape[0])]
        )
        threshold_dict["embs"].extend(embs[word].tolist())
    return threshold_dict


def run_clustering(args, threshold_dict, target_words, thresholds):
    for threshold in thresholds:
        logging.info(threshold)
        numbers_of_senses = []

        df = pd.DataFrame(threshold_dict)
        print(df.shape)
        df_unique = df.drop_duplicates(subset=[DEFS_COLUMN])
        print(df_unique.shape)
        clusters_dict = {}

        for target_word in tqdm(target_words):
            this_word = df_unique[df_unique[WORD_COLUMN]==target_word].reset_index()
            # sim_matrix = pairwise_distances(this_word.embs, this_word.embs, metric="cosine")
            # clusters = {}
            # indices = np.where(sim_matrix < threshold)
            # seen_col_numbers = set()
            # for i, definition in enumerate(this_word[DEFS_COLUMN]):
            #     if i not in seen_col_numbers:
            #         col_numbers = indices[1][np.where(indices[0]==i)]
            #         col_numbers = col_numbers[np.where(np.isin(col_numbers, list(seen_col_numbers), invert=True))]
            #         seen_col_numbers = seen_col_numbers.union(set(col_numbers))
            #         clusters[definition] = [this_word[DEFS_COLUMN][n] for n in col_numbers]
            #
            # numbers_of_senses.append(len(clusters))
            vectors = this_word.embs.to_list()
            linkage = "single"
            clustering = AgglomerativeClustering(
                n_clusters=None,
                distance_threshold=threshold,
                compute_full_tree=True,
                metric="cosine",
                linkage=linkage,
            ).fit(vectors)
            numbers_of_senses.append(clustering.n_clusters_)
            label2def = []
            # for label in np.unique(clustering.labels_):
            #     indices = np.where(clustering.labels_ == label)[0]
            #     defs_cluster = [defs.iloc[i] for i in indices]
            #     lengths = [len(definition) for definition in defs_cluster]
            #     longest = np.argmax(lengths)
            #     label2def[label] = defs_cluster[longest]

            # threshold_dict[SENSE_ID_COLUMN].extend([label2def[label] for label in clustering.labels_])
            #threshold_dict[SENSE_ID_COLUMN].extend(clustering.labels_.tolist())
            for definition, label in zip(this_word[DEFS_COLUMN], clustering.labels_):
                row_number = df[df[DEFS_COLUMN]==definition].index
                df.loc[row_number, SENSE_ID_COLUMN] = label
        avg_num_of_clusters = sum(numbers_of_senses)/len(numbers_of_senses)
        logging.info(avg_num_of_clusters)
        df.insert(2, SENSE_ID_COLUMN, df.pop(SENSE_ID_COLUMN))
        first = df[df[TIME_PERIOD_COLUMN]==1]
        first.to_csv(
            os.path.join(
                args.results_dir,
                f"{threshold}_{linkage}_corpus{1}.tsv",
            ),
            sep="\t",
            index=False,
            header=False,
        )
        second = df[df[TIME_PERIOD_COLUMN] == 2]
        second.to_csv(
            os.path.join(
                args.results_dir,
                f"{threshold}_{linkage}_corpus{2}.tsv",
            ),
            sep="\t",
            index=False,
            header=False,
        )


def main():
    args = parse_args()
    thresholds = [round(x, 2) for x in np.arange(0.6, 0.9, 0.1)]
    logging.info(f"Loading embeddings for time period 1")
    period_1, period_2 = 1, 2
    wordtable_1, embs_1 = load_np(args.path, period_1)
    logging.info(f"Loading embeddings for time period 2")
    wordtable_2, embs_2 = load_np(args.path, period_2)
    threshold_dict = {WORD_COLUMN: [], DEFS_COLUMN: [],  TIME_PERIOD_COLUMN: [], "embs": [] }
    target_words = wordtable_1[WORD_COLUMN].unique()
    threshold_dict = create_threshold_dict(target_words, threshold_dict, wordtable_1, period_1, embs_1)
    threshold_dict = create_threshold_dict(target_words, threshold_dict,
                                           wordtable_2, period_2, embs_2)

    run_clustering(args, threshold_dict, target_words, thresholds)

if __name__ == '__main__':
    main()
