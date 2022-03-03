# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import argparse
import os
import sys
from pathlib import Path

import cpc.feature_loader as fl
import progressbar
import torch
from cpc.cpc_default_config import set_default_cpc_config
from cpc.dataset import findAllSeqs
from cpc.train import loadCriterion, loadCriterionSNR, loadCriterionReverb


def parse_args(argv):
    parser = argparse.ArgumentParser(description='CPC + VAD inference script')
    parser = set_default_cpc_config(parser)
    parser.add_argument('--pathDB', type=str, required=True,
                        help="Path to the directory containing the audio data.")
    parser.add_argument('--pathPretrained', type=str, required=True,
                        help="Path to the pretrained VAD + snr + reverb predictor")
    parser.add_argument('--pathOut', type=str, required=True,
                        help="Path of the output directory where the "
                             "predictions will be saved.")
    parser.add_argument('--file_extension', type=str, default=".flac",
                        help="Extension of the audio files in pathDB.")
    parser.add_argument('--window_size', type=int, default=20480,
                        help="Number of frames to consider in each batch.")
    parser.add_argument('--no_sentence_level', action='store_true',
                        help='If activated, will return snr and reverb scores for each window of size window_size.'
                             'If not, will average those score and will return 1 score for each file.')
    parser.add_argument('--debug', action='store_true',
                        help='If activated, will load only a small number '
                             'of audio data.')
    parser.add_argument('--ignore_cache', action='store_true',
                        help="Activate if the sequences in pathDB have"
                             " changed.")
    args = parser.parse_args(argv)
    if args.pathPretrained[-3:] != '.pt':
        raise ValueError("--pathPretrained should point to a .pt file")
    args.pathPretrained = Path(args.pathPretrained).resolve()
    args.pathOut = Path(args.pathOut).resolve()
    return args


def write_predictions(out_path, predictions):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    torch.save(predictions, out_path)


def main(argv):
    args = parse_args(argv)
    args.pathOut = Path(args.pathOut).resolve()

    print("Inference mode...")
    # Find sequences
    seqNames, _ = findAllSeqs(args.pathDB,
                              extension=args.file_extension,
                              loadCache=not args.ignore_cache,
                              speaker_level=0)

    if args.debug:
        seqNames = seqNames[0:max(50, len(seqNames))]

    # Load model and criterion
    model, hidden_gar, hidden_encoder = fl.loadModel([args.pathPretrained],
                                                     loadStateDict=True)
    downsampling_factor = model.models[0].gEncoder.DOWNSAMPLING
    model.cuda()
    criterion_speech = loadCriterion(args, args.pathPretrained, downsampling_factor, None, 2, inference_mode=True).cuda()
    criterion_snr = loadCriterionSNR(args, args.pathPretrained, inference_mode=True).cuda()
    criterion_reverb = loadCriterionReverb(args, args.pathPretrained, inference_mode=True).cuda()

    # Load feature maker
    def get_predictions(x):
        return fl.inferVAD(model, x,
                           criterions=[criterion_speech, criterion_snr, criterion_reverb],
                           window_size=args.window_size,
                           downsampling_factor=downsampling_factor,
                           utt_level=not args.no_sentence_level)

    # Loop through files and extract features
    os.makedirs(args.pathOut, exist_ok=True)
    bar = progressbar.ProgressBar(maxval=len(seqNames))
    bar.start()
    for index, values in enumerate(seqNames):
        _, sub_path = values
        file_path = str(Path(args.pathDB) / sub_path)
        bar.update(index)
        predictions = get_predictions(file_path)
        out_path = args.pathOut / sub_path.replace(args.file_extension, '.pt')
        write_predictions(out_path, predictions)


if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn')
    args = sys.argv[1:]
    main(args)
