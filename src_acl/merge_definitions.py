import argparse
import csv
import logging
import os
import random

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
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
    parser.add_argument(
        '--results_dir',
        default=None,
        type=str,
        required=True,
        help='Results directory.',
    )
    parser.add_argument('--agglomerative', default=True)
    parser.add_argument('--linkage', default="average")
    parser.add_argument('--metric', default="cosine")
    parser.add_argument('--distances_range_start', default=0.1, type=float)
    parser.add_argument('--distances_range_end', default=1.0, type=float)
    return parser.parse_args()


def load_np(path, period):
    wordtable = pd.read_csv(
        os.path.join(path, f"definitions.english{period}.tsv"),
        sep='\t',
        quoting=csv.QUOTE_NONE,
    )
    embs = np.load(
        open(os.path.join(path, f"embeddings_perword.english{period}.npz"),
             "rb"))
    return wordtable, embs


def create_threshold_dict(target_words, threshold_dict, wordtable, period,
                          embs):
    for word in target_words:
        this_word = wordtable[wordtable[WORD_COLUMN] == word]
        defs = this_word[DEFS_COLUMN]
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


def write_results(df, args, metric, numbers_of_senses):
    avg_num_of_clusters = sum(numbers_of_senses) / len(
        numbers_of_senses)
    logging.info(avg_num_of_clusters)
    df.insert(2, SENSE_ID_COLUMN, df.pop(SENSE_ID_COLUMN))
    first = df[df[TIME_PERIOD_COLUMN] == 1]
    first.to_csv(
        os.path.join(
            args.results_dir,
            f"{args.linkage}_{metric}corpus{1}.tsv",
        ),
        sep="\t",
        index=False,
        header=False,
    )
    second = df[df[TIME_PERIOD_COLUMN] == 2]
    second.to_csv(
        os.path.join(
            args.results_dir,
            f"{args.linkage}_{metric}corpus{2}.tsv",
        ),
        sep="\t",
        index=False,
        header=False,
    )


def run_clustering(args, threshold_dict, target_words, thresholds):
    numbers_of_senses_silhouette = []

    df = pd.DataFrame(threshold_dict)
    df_unique = df.drop_duplicates(subset=[DEFS_COLUMN])

    for target_word in tqdm(target_words):
        sils = []
        results = []
        this_word = df_unique[
            df_unique[WORD_COLUMN] == target_word
            ].reset_index()
        for threshold in thresholds:
            logging.info(threshold)

            if args.agglomerative:
                vectors = this_word.embs.to_list()
                clustering = AgglomerativeClustering(
                    n_clusters=None,
                    distance_threshold=threshold,
                    compute_full_tree=True,
                    metric=args.metric,
                    linkage=args.linkage,
                ).fit(vectors)

                try:
                    sils.append(silhouette_score(vectors, clustering.labels_,
                                                 metric='cosine'))
                except ValueError:
                    logging.info("Unable to calculate metrics 1 label only")
                    sils.append(-1)
                results.append((clustering.n_clusters_, clustering.labels_))
        best_silhouette = np.argmax(sils)
        numbers_of_senses_silhouette.append(results[best_silhouette][0])

        for definition, label in zip(
                this_word[DEFS_COLUMN],
                results[best_silhouette][1],
        ):
            row_number = df[df[DEFS_COLUMN] == definition].index
            df.loc[row_number, SENSE_ID_COLUMN] = label

    write_results(df, args, "Silhouette", numbers_of_senses_silhouette)


def main():
    args = parse_args()
    thresholds = [
        round(x, 2) for x in np.arange(
            args.distances_range_start,
            args.distances_range_end,
            0.1,
        )
    ]
    logging.info(f"Loading embeddings for time period 1")
    period_1, period_2 = 1, 2
    wordtable_1, embs_1 = load_np(args.path, period_1)
    logging.info(f"Loading embeddings for time period 2")
    wordtable_2, embs_2 = load_np(args.path, period_2)
    threshold_dict = {WORD_COLUMN: [], DEFS_COLUMN: [], TIME_PERIOD_COLUMN: [],
                      "embs": []}
    target_words = wordtable_1[WORD_COLUMN].unique()
    threshold_dict = create_threshold_dict(target_words, threshold_dict,
                                           wordtable_1, period_1, embs_1)
    threshold_dict = create_threshold_dict(target_words, threshold_dict,
                                           wordtable_2, period_2, embs_2)

    run_clustering(args, threshold_dict, target_words, thresholds)


if __name__ == '__main__':
    main()
