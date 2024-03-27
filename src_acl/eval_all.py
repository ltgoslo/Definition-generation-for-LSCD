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
            f"data/{language}/truth/graded.txt",
        ], capture_output=True,
    )
    value = result.stdout.decode('utf8').split("\t")
    score, p = value[-2], value[-1].rstrip()
    out = f"{language} {strat} {metric}: score {score} p-value {p}"
    if float(value[-1]) > 0.05:  # yes this is rough
        # but probably enough for us
        out += " too large p-value :("
    f.write(f"{out}\n")


def main():
    with open("result.txt", "w", encoding="utf8") as f:
        for language in tqdm(LANGS):
            for strat_folder in glob(f"merge_results/{language}/*"):
                run("Cosine", strat_folder, language, f)
                run("JS", strat_folder, language, f)


if __name__ == '__main__':
    main()
