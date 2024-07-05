#! /usr/bin/env python3

import sys, csv, string, re
import pandas as pd
import networkx as nx
import Levenshtein
import gzip
import argparse


def load2PeriodsData(period1_file, period2_file):
    df1 = pd.read_csv(period1_file, delimiter="\t", quoting=csv.QUOTE_NONE, encoding="utf-8", compression="infer", header=None, names=["Word", "Context_Prompt", "Definition"])
    df2 = pd.read_csv(period2_file, delimiter="\t", quoting=csv.QUOTE_NONE, encoding="utf-8", compression="infer", header=None, names=["Word", "Context_Prompt", "Definition"])
    df1_agg = df1.groupby(["Word", "Definition"]).size().reset_index(name='Count_P1')
    df2_agg = df2.groupby(["Word", "Definition"]).size().reset_index(name='Count_P2')
    df_all = pd.merge(df1_agg, df2_agg, how="outer").fillna(0)
    df_all["Count_All"] = df_all["Count_P1"] + df_all["Count_P2"]
    return df1, df2, df_all


def load1PeriodData(period_file):
    df = pd.read_csv(period_file, delimiter="\t", quoting=csv.QUOTE_NONE, encoding="utf-8", compression="infer", header=None, names=["Word", "Context_Prompt", "Definition"])
    df_agg = df.groupby(["Word", "Definition"]).size().reset_index(name='Count_All')
    return df, df_agg


def preprocess(s):
    s = s.translate(str.maketrans('', '', string.punctuation))
    s = re.sub(r'\s+', ' ', s)
    s = s.lower().strip()
    return s

def createGraph(word_df):
    G = nx.Graph()
    for index, row in word_df.iterrows():
        #G.add_node(row.Definition, count=row.Count_All, p1count=row.Count_P1, p2count=row.Count_P2, norm=preprocess(row.Definition))
        G.add_node(row.Definition, count=row.Count_All, norm=preprocess(row.Definition))
    print("Create graph: {} nodes, {} edges, {} components".format(G.number_of_nodes(), G.number_of_edges(), nx.number_connected_components(G)))
    return G


def connectIdenticals(G):
    for node1, node1attr in G.nodes(data=True):
        for node2, node2attr in G.nodes(data=True):
            # tradeoff between has_edge and has_path - it's cheaper in most cases to add unnecessary edges rather than check if they are necessary or not
            if node1 == node2 or G.has_edge(node1, node2):
                continue
            if node1attr["norm"] == node2attr["norm"]:
                G.add_edge(node1, node2, method="idnorm")
    print("Connect identicals: {} nodes, {} edges, {} components".format(G.number_of_nodes(), G.number_of_edges(), nx.number_connected_components(G)))


# x is assumed to be the shorter string
def is_subseq(x, y):
    it = iter(y)
    return all(c in it for c in x)
## Problem: this should really be directional, otherwise completely unrelated strings end up with each other:
## e.g. B is substring of AB, A is substring of AB, therefore A and B are in the same component

def connectSubsequence(G, minLength=3):
    for node1, node1attr in G.nodes(data=True):
        for node2, node2attr in G.nodes(data=True):
            if node1 == node2 or G.has_edge(node1, node2):
                continue
            tokens1 = node1attr["norm"].split(" ")
            tokens2 = node2attr["norm"].split(" ")
            if len(tokens1) < minLength or len(tokens2) < minLength:
                continue
            if is_subseq(tokens1, tokens2) and len(tokens1) > len(tokens2)/2:
                G.add_edge(node1, node2, method="subseq")
    print("Connect subsequence: {} nodes, {} edges, {} components  (min length: {})".format(G.number_of_nodes(), G.number_of_edges(), nx.number_connected_components(G), minLength))


def keep_levenshtein(a, b, maxThreshold):
    if maxThreshold >= 1:
        dist = Levenshtein.distance(a, b)
        return dist < maxThreshold
    else:
        dist = Levenshtein.distance(a, b)
        normdist = dist / max([len(a), len(b)])
        return normdist < maxThreshold

def connectLevenshtein(G, maxThreshold=0.25):
    blacklist = {}
    for node1, node1attr in G.nodes(data=True):
        for node2, node2attr in G.nodes(data=True):
            if node1 == node2 or G.has_edge(node1, node2):
                continue
            # Levenshtein distances are symmetric, so we only need to compute them once
            if (node1, node2) in blacklist:
                continue
            if keep_levenshtein(node1attr["norm"], node2attr["norm"], maxThreshold):
                G.add_edge(node1, node2, method="levenshtein")
            else:
                blacklist[node2, node1] = True
    print("Connect Levenshtein: {} nodes, {} edges, {} components  (threshold: {})".format(G.number_of_nodes(), G.number_of_edges(), nx.number_connected_components(G), maxThreshold))
    #print("Blacklist length:", len(blacklist))


# maxFreq seems to give slightly better results than maxDegree, but it's hard to judge
def selectBest(G, method="maxFreq"):
    rewriteDict = {}
    ndefs = 0
    for C in nx.connected_components(G):
        S = G.subgraph(C).copy()
        if S.number_of_nodes() == 1:
            # if the definition doesn't change, we don't have to add it to the dictionary
            continue
        elif method == "maxFreq":
            maxFreq = max([(node, nodeattr["count"]) for node, nodeattr in S.nodes(data=True)], key=lambda x: x[1])
            bestDef = maxFreq[0]
        elif method == "maxDegree":
            maxDegree = max(dict(S.degree()).items(), key=lambda x: x[1])
            bestDef = maxDegree[0]
        else:
            print(f"Method {maxFreq} not defined!")
            break
        for node, nodeattr in S.nodes(data=True):
            rewriteDict[node] = bestDef
        ndefs += 1
    return rewriteDict, ndefs


def merge(df_all, words, args):
    print("Instances:", df_all["Count_All"].sum())
    print("Unique definitions before merging:", df_all.shape[0])
    rewrites = {}
    ndefs_all = 0
    # this should be relatively easy to parallelize if necessary...
    for word in sorted(words):
        print(word)
        G = createGraph(df_all[df_all["Word"]==word])
        connectIdenticals(G)
        if args.min_subseq_len >= 0:
            connectSubsequence(G, minLength=args.min_subseq_len)
        if args.leven_threshold >= 0:
            connectLevenshtein(G, maxThreshold=args.leven_threshold)
        rewrites[word], ndefs = selectBest(G, method=args.hub_strategy)
        ndefs_all += ndefs
    print("Unique definitions after merging:", ndefs_all)
    return rewrites


def writeResults(df, rewrites, outfilename):
    with gzip.open(outfilename, 'wt') as f:
        for i, row in df.iterrows():
            # Target word | Original definition | Merged definition
            # The sense_dis script expects the relevant definition in the 3rd column
            new_row = [row.Word, row.Definition, rewrites[row.Word].get(row.Definition, row.Definition)]
            f.write("\t".join(new_row) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--defgen_path",
        type=str,
        help="Directory with two files containing the generated definitions",
        required=True
    )
    parser.add_argument(
        "--lang",
        type=str,
        default="english",
        help="Task name (e.g. norwegian1)",
    )
    parser.add_argument(
        "--separate",
        action="store_true",
        help="Process both time periods separately (default: process them jointly)"
    )
    parser.add_argument(
        "--leven_threshold",
        type=float,
        help="Levenshtein threshold - only merge definitions below threshold. Uses relative Levenshtein distance with values <1 and absolute Levenshtein distance with values >=1. Set to <0 to skip this step.",
        default=0.25
    )
    parser.add_argument(
        "--min_subseq_len",
        type=int,
        help="Minimal length of the definition (in words) to be taken into account for subsequence matching. Set to <0 to skip this step.",
        default=3,
    )
    parser.add_argument(
        "--hub_strategy",
        choices=["maxFreq", "maxDegree"],
        help="Strategy for choosing the target definition of a subgraph",
        default="maxFreq"
    )
    parser.add_argument(
        "--out_path",
        type=str,
        help="Directory to save datasets with merged definitions",
        required=True
    )
    args = parser.parse_args()

    print("**************")
    print(args.lang.upper())
    print("**************")

    if args.separate:
        for period in ("1", "2"):
            print("Period", period)
            df, df_all = load1PeriodData(f"{args.defgen_path}/{args.lang}-corpus{period}.tsv.gz")
            words = df_all["Word"].unique()
            rewrites = merge(df_all, words, args)
            writeResults(df, rewrites, f"{args.out_path}/{args.lang}-corpus{period}.tsv.gz")
    else:
        df1, df2, df_all = load2PeriodsData(f"{args.defgen_path}/{args.lang}-corpus1.tsv.gz", f"{args.defgen_path}/{args.lang}-corpus2.tsv.gz")
        words = df_all["Word"].unique()
        rewrites = merge(df_all, words, args)
        writeResults(df1, rewrites, f"{args.out_path}/{args.lang}-corpus1.tsv.gz")
        writeResults(df2, rewrites, f"{args.out_path}/{args.lang}-corpus2.tsv.gz")
