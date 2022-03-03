from pathlib import Path
import argparse
import sys
import json
from common_voice_db_maker import savePhoneDict


def parse_ctc_labels_from_root(root, letters_path):
    letter2index = {}
    index2letter = {}

    with open(letters_path, "r") as f:
        for i, line in enumerate(f.readlines()):
            line = line.strip()

            letter2index[line] = i
            index2letter[i] = line

    result = {}

    for file in Path(root).rglob("*.trans.txt"):
        with open(file, "r") as f:
            for line in f.readlines():
                line = line.rstrip()
                p = line.find(" ")
                assert p > 0
                fname = line[:p]

                chars = line[p + 1 :].replace(" ", "|").lower()
                decoded = []

                for c in chars:
                    decoded.append(letter2index[c])
                result[fname] = decoded

    return result, index2letter


def parse_args(argv):

    parser = argparse.ArgumentParser(description="Character DB maker")
    parser.add_argument("path_input", type=str)
    parser.add_argument("path_letters", type=str)
    parser.add_argument("output", type=str)

    return parser.parse_args(argv)


if __name__ == "__main__":

    args = parse_args(sys.argv[1:])
    char_dict, index2letter = parse_ctc_labels_from_root(
        args.path_input, args.path_letters
    )

    args.output = Path(args.output)
    args.output.mkdir(exist_ok=True)

    savePhoneDict(char_dict, args.output / f"{args.output.stem}_chars.txt")
    path_out_index = args.output / "char_index.json"
    with open(path_out_index, "w") as file:
        json.dump(index2letter, file, indent=2)
