import os
import sys
import argparse
from pathlib import Path
from progressbar import ProgressBar
from cpc.dataset import findAllSeqs, filterSeqs


def move_sym_db(path_in, path_out, seq_list, prefix_out):

    bar = ProgressBar(maxval=len(seq_list))
    bar.start()

    path_in = Path(path_in)
    path_out = Path(path_out)

    for index, loc_path in enumerate(seq_list):
        bar.update(index)

        parts = list(Path(loc_path).parts)
        parts[0] = f"{prefix_out}_{parts[0]}"
        parent_outs = Path(*parts[:-1])
        name_out = f"{prefix_out}_{Path(loc_path).stem}{Path(loc_path).suffix}"
        new_path = path_out / parent_outs / name_out

        Path(new_path).parent.mkdir(exist_ok=True, parents=True)
        os.symlink(str(path_in / loc_path), str(new_path))

    bar.finish()


def get_seq_list(path_list_in, prefix_out):

    with open(path_list_in, "r") as file:
        names_in = [p.strip() for p in file.readlines()]

    return [f"{prefix_out}_{x}" for x in names_in]


def merge_seq_lists(seq_lists, names_in, path_out):

    out = []
    for seq_list, name in zip(seq_lists, names_in):
        out += get_seq_list(seq_list, name)

    with open(path_out, "w") as file:
        for seq_name in out:
            file.write(seq_name + "\n")


def parse_args(argv):

    parser = argparse.ArgumentParser(description="Dataset merge")

    subparsers = parser.add_subparsers(dest="command")
    parser_db = subparsers.add_parser("cp_db")
    parser_db.add_argument("path_dbs", type=str, nargs="*")
    parser_db.add_argument("-o", "--output", type=str, required=True)
    parser_db.add_argument("--file_extension", type=str, default=".flac")
    parser_db.add_argument("--names", type=str, default=None, nargs="*")

    parser_filters = subparsers.add_parser("merge_subsets")
    parser_filters.add_argument("path_subsets", type=str, nargs="*")
    parser_filters.add_argument("--names", type=str, default=None, nargs="*")
    parser_filters.add_argument("-o", "--output", type=str, required=True)

    return parser.parse_args(argv)


if __name__ == "__main__":

    args = parse_args(sys.argv[1:])

    if args.command == "cp_db":
        for i_, path_db in enumerate(args.path_dbs):
            seq_list = [
                x[1] for x in findAllSeqs(path_db, extension=args.file_extension)[0]
            ]
            if args.names is not None:
                prefix_out = args.names[i_]
            else:
                prefix_out = str(i_)
            move_sym_db(path_db, args.output, seq_list, prefix_out)
    elif args.command == "merge_subsets":

        if args.names is None:
            args.names = list(range(len(args.path_subsets)))
        elif len(args.names) != len(args.path_subsets):
            raise RuntimeError(
                "The number of names provided should match the number of subsets"
            )

        merge_seq_lists(args.path_subsets, args.names, args.output)
