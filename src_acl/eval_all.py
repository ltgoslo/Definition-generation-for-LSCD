from glob import glob
import os
import subprocess
import sys

from tqdm import tqdm

LANGS = (
    "english",
    "russian1",
    "russian2",
    "russian3",
    "norwegian1",
    "norwegian2"
)


def run(metric, strat_folder, language, f):
    strat = os.path.split(strat_folder)[-1]
    result = subprocess.run(
        [
            sys.executable,
            "../src/eval.py",
            "2",
            f"{strat_folder}/defgen/{language}/{metric}_dict.tsv",
            f"../src/data/{language}/truth/graded.txt", # the ground truth on saga had landet changed to land . omg
        ], capture_output=True,
    )
    value = result.stdout.decode('utf8').split("\t")
    score, p = value[-2], value[-1].rstrip()
    out = f"{language} {strat} {metric}: score {score} p-value {p}"
    if float(value[-1]) > 0.05:  # yes this is rough
        # but probably enough for us
        out += " too large p-value :("
    f.write(f"{out}\n")


def run_lesk(strat_folder, language, f):
    strat = "Lesk without PoS"
    if "pos_True" in strat_folder:
        strat = "Lesk with PoS"
    for metric_dict in glob(f"{strat_folder}/*_dict.tsv"):
        result = subprocess.run(
            [
                sys.executable,
                "../src/eval.py",
                "2",
                metric_dict,
                f"../../acl_data/{language}/truth/graded.txt",
            ], capture_output=True,
        )
        value = result.stdout.decode('utf8').split("\t")
        score, p = value[-2], value[-1].rstrip()
        metric = os.path.split(metric_dict)[-1].replace("_dict.tsv", "")
        out = f"{language} {strat} {metric}: score {score} p-value {p}"
        if float(value[-1]) > 0.05:  # yes this is rough
            # but probably enough for us
            out += " too large p-value :("
        f.write(f"{out}\n")


def main():
    pred_path = "predictions/"
    with open("result.txt", "w", encoding="utf8") as f:
        for language in tqdm(LANGS):
            for strat_folder in glob(f"{pred_path}merge_results/{language}/*"):
                run("Cosine", strat_folder, language, f)
                run("JS", strat_folder, language, f)
            for lesk_folder in glob(f"{pred_path}lesk/{language}*"):
                run_lesk(lesk_folder, language, f)


if __name__ == '__main__':
    main()
