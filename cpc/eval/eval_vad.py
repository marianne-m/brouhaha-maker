import sys
import argparse
from config import *
from torch import nn as nn
from cpc.eval.scorer.vad_cpc import ScorerVadCpc
from cpc.eval.scorer.vad_pyannote import ScorerVadPya
from cpc.eval.scorer.snr_cpc import ScorerSnrCpc
from cpc.eval.scorer.reverb_cpc import ScorerReverbCpc


def parse_args(argv):
    parser = argparse.ArgumentParser(description='Prepare a dataset.')
    parser.add_argument('--dataset', type=str, required=True,
                        choices=['allstar','ava','buckeye','coraal','librispeech','voxpopuli'],
                        help='Dataset we want to work with')
    parser.add_argument('--path_pred', type=str, required=True,
                        help='Path to the predictions of our model')
    parser.add_argument('--overlap', type=int, required=False, default=1,
                        help='Number of predictions made for each frame when using'
                        'sliding windows: window_overlap = size_window / overlap.')
    parser.add_argument('--batch_size', type=int, required=False, default=8,
                        help='Size of prediction batches.')
    parser.add_argument('--sequence_size', type=int, required=False, default=128,
                        help='Size of prediction sequences.')
    parser.add_argument('--action', type=str, required=True,
                        choices=['vad_cpc', 'snr_cpc', 'reverb_cpc', 'vad_pya'], nargs='*',
                        help='Score we want to compute for our dataset')    
    parser.add_argument('--m', type=str, required=False, default='',
                        help='Comment describing the score to compute')                  
    args = parser.parse_args(argv)
    return args


def main(argv):
    args = parse_args(argv)

    dataset_dict = {'allstar': 'ALLSTAR_16k',
                    'ava': 'AVA',
                    'buckeye': 'buckeye',
                    'coraal': 'CORAAL_16k',
                    'librispeech': 'LibriSpeech',
                    'voxpopuli': 'VoxPopuli/fr'}

    # evaluate a CPC model for VAD
    if 'vad_cpc' in args.action:
        print(f"Compute the performance of CPC for VAD on {args.dataset}")
        scorer_cpc = ScorerVadCpc(dataset_dict[args.dataset], args.path_pred, args.overlap, args.batch_size, args.sequence_size)
        scorer_cpc(args.m)

    # evaluate a pyannote model for VAD
    if 'vad_pya' in args.action:
        print(f"Compute the performance of Pyannote for VAD on {args.dataset}")
        scorer_pya = ScorerVadPya(dataset_dict[args.dataset], args.path_pred)
        scorer_pya(args.m)

    # evaluate a CPC model for SNR prediction
    if 'snr_cpc' in args.action:
        print(f"Compute the performance of CPC for SNR prediction on {args.dataset}")
        scorer_snr = ScorerSnrCpc(dataset_dict[args.dataset], args.path_pred, args.overlap, args.batch_size, args.sequence_size)
        scorer_snr(args.m)

    # evaluate a CPC model for reverb prediction
    if 'reverb_cpc' in args.action:
        print(f"Compute the performance of CPC for Reverb prediction on {args.dataset}")
        scorer_snr = ScorerReverbCpc(dataset_dict[args.dataset], args.path_pred, args.overlap, args.batch_size, args.sequence_size)
        scorer_snr(args.m)


if __name__ == "__main__":
    # execute only if run as a script
    args = sys.argv[1:]
    main(args)
    
        

