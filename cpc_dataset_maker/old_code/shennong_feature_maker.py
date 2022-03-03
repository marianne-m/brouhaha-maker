import os
import numpy as np
import progressbar
import torchaudio
import argparse
import sys
from shennong.audio import Audio
from shennong.features.processor.bottleneck import BottleneckProcessor
from multiprocessing import Process, Lock, Value


def makeFeature(pathIn, pathOut, processor):
    audio, sampleRate = torchaudio.load(pathIn)
    RATIO = 32857
    audio *= RATIO
    # audio= Audio.load(pathIn)
    audio = Audio(audio.view(-1, 1).short().numpy(), sampleRate)
    features = processor.process(audio)
    with open(pathOut, "wb") as f:
        np.save(f, features.data)


def makeFeaturesFromList(pathDB, filesNames, pathOut, processor):

    nItems = len(filesNames)
    print(f"{nItems} files found")
    bar = progressbar.ProgressBar(nItems)
    bar.start()

    if not os.path.isdir(pathOut):
        os.mkdir(pathOut)

    for index, name in enumerate(filesNames):
        bar.update(index)
        pathFileIn = os.path.join(pathDB, name)
        pathFileOut = os.path.join(pathOut, f"{os.path.splitext(name)[0]}.npy")
        makeFeature(pathFileIn, pathFileOut, processor)

    bar.finish()


def makeFeaturesFromListMultProc(pathDB, filesNames, pathOut, processor, nProcess=10):

    nItems = len(filesNames)
    print(f"{nItems} files found")
    bar = progressbar.ProgressBar(nItems)
    bar.start()

    if not os.path.isdir(pathOut):
        os.mkdir(pathOut)

    def run(v, l, indexStart, indexEnd):
        if indexStart > len(filesNames):
            return
        data = filesNames[indexStart:indexEnd]
        for name in data:
            pathFileIn = os.path.join(pathDB, name)
            pathFileOut = os.path.join(pathOut, f"{os.path.splitext(name)[0]}.npy")
            makeFeature(pathFileIn, pathFileOut, processor)
            l.acquire()
            v.value += 1
            bar.update(v.value)
            l.release()

    stack = []
    sizeSlice = (len(filesNames) // nProcess) + 1
    lock = Lock()
    v = Value("i", 0)
    for i in range(nProcess):
        indexStart = sizeSlice * i
        indexEnd = indexStart + sizeSlice if i < nProcess - 1 else len(filesNames)
        p = Process(target=run, args=(v, lock, indexStart, indexEnd))
        p.start()
        stack.append(p)

    for process in stack:
        p.join()

    bar.finish()


def parse_args(argv):
    parser = argparse.ArgumentParser(description="Shennong feature maker")
    parser.add_argument("pathDB", type=str)
    parser.add_argument("pathOut", type=str)
    parser.add_argument(
        "--feature_mode", type=str, choices=["Bottleneck"], default="Bottleneck"
    )
    parser.add_argument("--file_extension", type=str, default=".mp3")
    return parser.parse_args(argv)


def main(argv):

    args = parse_args(argv)

    if args.feature_mode == "Bottleneck":
        processor = BottleneckProcessor(weights="BabelMulti")
    else:
        raise ValueError(f"Invalid feature's name {args.feature_mode}")

    listFiles = [
        f
        for f in os.listdir(args.pathDB)
        if os.path.splitext(f)[1] == args.file_extension
    ]

    makeFeaturesFromList(args.pathDB, listFiles, args.pathOut, processor)
    # makeFeaturesFromListMultProc(args.pathDB, listFiles, args.pathOut, processor)


if __name__ == "__main__":
    main(sys.argv[1:])
