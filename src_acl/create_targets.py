import argparse
import os


def parse_arge():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--lang",
        default="norwegian1",
    )
    parser.add_argument(
        "--targets_path",
        default="../../gloss-annotator/wugs/nor_dia_change/subset1/data",
    )
    parser.add_argument(
        "--res_path",
        default="../../acl_data/"
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arge()
    if "norwegian" in args.lang:
        targets = os.listdir(args.targets_path)
        if args.lang == "norwegian1":
            for wrong_lemma in {"formiddagen", "landet"}:
                i = targets.index(wrong_lemma)
                targets[i] = wrong_lemma[:-2]
    with open(
            os.path.join(args.res_path, args.lang, "targets.txt"),
            "w",
            encoding="utf8",
    ) as res_file:
        for target in targets:
            res_file.write(target + "\n")
