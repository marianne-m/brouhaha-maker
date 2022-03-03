import argparse
import random
import matplotlib.pyplot as plt
import numpy as np
import sys
import math
import utils.sequence_data as sd
import utils.samplers as samplers
from pathlib import Path
from utils.plot_stats import plot_hist, plot_as_hist, plot_edge_hist


def load_group(path_stats_dir, max_n_seqs=5000):
    path_cache_sequence = str(Path(path_stats_dir) / "_cache.txt")
    path_cache_speaker = str(Path(path_stats_dir) / "_cache_speaker.txt")
    sequence_data = sd.load_sequence_data(path_cache_sequence)

    if max_n_seqs < len(sequence_data):
        sequence_data = random.choices(sequence_data, k=max_n_seqs)
    return sequence_data


def parse_args(argv):

    parser = argparse.ArgumentParser(
        description="Compare the statistics of different datasets"
    )
    parser.add_argument("path_stats_dirs", type=str, nargs="*")
    parser.add_argument("-o", "--output", type=str, default="coin")
    parser.add_argument("--max_n_seqs", type=int, default=15000)
    parser.add_argument("--to_log", action="store_true")
    parser.add_argument("--names", type=str, nargs="*", default=None)

    return parser.parse_args(argv)


def main(args):

    sequence_groups = []

    if args.names is not None:
        assert len(args.names) == len(args.path_stats_dirs)

    for i_, stat_dir in enumerate(args.path_stats_dirs):

        print(stat_dir)
        seq_data = load_group(stat_dir, args.max_n_seqs)
        seq_name = Path(stat_dir).stem
        if args.names is not None:
            seq_name = args.names[i_]
        sequence_groups.append((seq_name, seq_data))

    args.path_out = Path(args.output)
    args.path_out.mkdir(exist_ok=True)
    default_output = ["perplexity"]

    plt.rc("xtick", labelsize=15)
    plt.rc("ytick", labelsize=15)
    for attr in default_output:
        fig, ax = plt.subplots(tight_layout=True)
        for name, group in sequence_groups:

            data = [getattr(x, attr) for x in group]
            if args.to_log:
                data = [math.log(x) for x in data]
            plot_edge_hist(ax, data, name, 100)
        ax.legend(fontsize="xx-large")
        fig.savefig(str(Path(args.output) / f"{attr}_stats.png"))


if __name__ == "__main__":
    argv = sys.argv[1:]
    args = parse_args(argv)
    main(args)
