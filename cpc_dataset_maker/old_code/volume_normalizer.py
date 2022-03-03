import os
import argparse
import torchaudio
import progressbar
import math
import sys

# import sox
import torch
from random import shuffle
from cpc.dataset import findAllSeqs

from multiprocessing import Pool
import tqdm


def applyGainAndNorm(args):
    pathDB, pathOutput, seqName, gain, normalize = args
    fullPath = os.path.join(pathDB, seqName)
    fullPathOut = os.path.join(pathOutput, seqName)
    dirOut = os.path.dirname(fullPathOut)
    if not os.path.isdir(dirOut):
        os.makedirs(dirOut, exist_ok=True)
    try:
        seq, sr = torchaudio.load(fullPath)
    except RuntimeError:
        return

    if normalize:
        max_abs = seq.abs().max()
        seq = seq / max_abs
    seq = torch.clamp(seq * gain, -1.0, 1.0)
    torchaudio.save(fullPathOut, seq, sr)


def applyGainToDB(args, inSeqs):

    tasks = [
        (args.pathDB, args.pathOutput, seq_name, args.gain, args.normalize)
        for seq_name in inSeqs
    ]
    with Pool(processes=args.n_processes) as pool:
        for _ in tqdm.tqdm(
            pool.imap_unordered(applyGainAndNorm, tasks, chunksize=5), total=len(tasks)
        ):
            pass


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Volume normalizer")
    subparsers = parser.add_subparsers(dest="command")

    parser_gain = subparsers.add_parser("gain")
    parser_gain.add_argument("pathDB", type=str)
    parser_gain.add_argument("pathOutput", type=str)
    parser_gain.add_argument("gain", type=float)
    parser_gain.add_argument("--file_extension", type=str, default=".wav")
    parser_gain.add_argument("--normalize", action="store_true")
    parser_gain.add_argument("-j", "--n_processes", type=int, default=16)

    parser_inspect = subparsers.add_parser("inspect")
    parser_inspect.add_argument("pathDB", type=str)
    parser_inspect.add_argument("--dataset_levels", type=int, default=1)
    parser_inspect.add_argument("--file_extension", type=str, default=".wav")

    args = parser.parse_args()

    inSeqs = [x[1] for x in findAllSeqs(args.pathDB, extension=args.file_extension)[0]]

    if args.command == "gain":
        applyGainToDB(args, inSeqs)
    elif args.command == "inspect":
        shuffle(inSeqs)
        inSeqs = inSeqs[:1000]
        bar = progressbar.ProgressBar(maxval=len(inSeqs))
        bar.start()

        m_min, m_max = 0, 0
        v_min, v_max = 0, 0
        for index, seqName in enumerate(inSeqs):
            bar.update(index)

            fullPath = os.path.join(args.pathDB, seqName)
            data = torchaudio.load(fullPath)[0]

            m_min += data.min().item()
            m_max += data.max().item()
            v_min += data.min().item() * data.min().item()
            v_max += data.max().item() * data.max().item()

        bar.finish()
        nItems = len(inSeqs)
        m_min /= nItems
        m_max /= nItems
        v_min = v_min / nItems - m_min * m_min
        v_max = v_max / nItems - m_max * m_max

        print(f"Average min : {m_min}, std min {math.sqrt(max(v_min, 0))}")
        print(f"Average max : {m_max}, std max {math.sqrt(max(v_max, 0))}")
