from glob import glob
import gzip
import os
import re

PATTERN = re.compile(r"What is the definition of (\w+)\?")

if __name__ == '__main__':
    data_path = "../acl_results"
    res_path = os.path.join(data_path, "fixed")
    if not os.path.exists(res_path):
        os.mkdir(res_path)
    for corpora in glob(f"{data_path}/*.txt"):
        fn = corpora.split("/")[-1]
        res_path_file = os.path.join(res_path, fn.replace("-home-m-corpora-acl-parsed-english-", "").replace(".conllu.gz.txt.gz.txt", "") + ".tsv.gz")
        with gzip.open(res_path_file, "wt") as results_file:
            with open(corpora, "r", encoding="utf8") as f:
                for line in f:
                    prompt, definition = line.split("\t")
                    target = re.search(PATTERN, prompt).group(1)
                    results_file.write(f"{target}\t{prompt}\t{definition}")
