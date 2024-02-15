import argparse
import os


def parse_arge():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--lang",
        default="norwegian1",
    )
    parser.add_argument(
        "--data_dir",
        default="../../acl_data",
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arge()
    targets = []

    with open(f"{args.data_dir}/{args.lang}/truth/graded.txt") as f:
        lines = f.readlines()
        for el in lines:
            splitted = el.split()
            if splitted[0] not in {"formiddagen", "landet"}:
                targets.append(splitted[0])
            else:
                targets.append(splitted[0][:-2])
    with open(
            os.path.join(args.data_dir, args.lang, "targets.txt"),
            "w",
            encoding="utf8",
    ) as res_file:
        for target in targets:
            res_file.write(target + "\n")
