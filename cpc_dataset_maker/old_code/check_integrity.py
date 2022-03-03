import argparse
import sys
from pathlib import Path


def load_names(path_names):
    with open(path_names, "r") as file:
        return [x.strip() for x in file.readlines()]


def chec_integrity(group_1, group_2):

    d_1_ = set([Path(x).stem for x in group_1])
    d_2_ = set([Path(x).stem for x in group_2])

    return len(d_1_.intersection(d_2_)) == 0


def parse_args(argv):

    parser = argparse.ArgumentParser(description="Integrity check")
    parser.add_argument("path_subsets", type=str, nargs="*")
    return parser.parse_args(argv)


if __name__ == "__main__":

    args = parse_args(sys.argv[1:])
    n_invalid = 0

    for index_1, name_1 in enumerate(args.path_subsets):
        group_1 = load_names(name_1)
        for name_2 in args.path_subsets[index_1 + 1 :]:

            group_2 = load_names(name_2)
            integrity_status = chec_integrity(group_1, group_2)

            if not integrity_status:
                print(f"{name_1} and {name_2} have common data")
                n_invalid += 1

    if n_invalid == 0:
        print("OK")
    else:
        print(f"FAILED : {n_invalid} couples")
