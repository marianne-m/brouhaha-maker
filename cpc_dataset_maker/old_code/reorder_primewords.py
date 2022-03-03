import argparse
import json
import sys
from pathlib import Path
from progressbar import ProgressBar


def update_data_with_full_path(path_db, data_list):

    bar = ProgressBar(maxval=len(data_list))
    bar.start()

    for index, data_info in enumerate(data_list):
        bar.update(index)
        seq_name = Path(data_info["file"]).stem
        dir0 = seq_name[0]
        dir1 = seq_name[:2]

        data_info["full_path"] = Path(path_db) / dir0 / dir1 / data_info["file"]

    bar.finish()


def move_files(data_list, path_db_out):

    bar = ProgressBar(maxval=len(data_list))
    bar.start()

    for index, data_info in enumerate(data_list):
        bar.update(index)
        path_in = Path(data_info["full_path"])
        path_out = Path(path_db_out) / data_info["user_id"] / path_in.name
        try:
            path_in.replace(path_out)
        except FileNotFoundError:
            continue

    bar.finish()


def main(args):

    with open(args.path_transcript, "rb") as file:
        data = json.load(file)

    Path(args.output).mkdir(exist_ok=True)

    update_data_with_full_path(args.path_db, data)

    speakers = {x["user_id"] for x in data}
    for speaker in speakers:
        (Path(args.output) / speaker).mkdir(exist_ok=True)
    move_files(data, args.output)


def parse_args(argv):

    parser = argparse.ArgumentParser(description="Dataset statistics")

    parser.add_argument("path_db", type=str)
    parser.add_argument("path_transcript")
    parser.add_argument("--file_extension", type=str, default=".wav")
    parser.add_argument("-o", "--output", type=str, default="coin")

    return parser.parse_args(argv)


if __name__ == "__main__":

    argv = sys.argv[1:]
    args = parse_args(argv)
    main(args)
