import os
import sys
import torch
import argparse
import pandas as pd
from tqdm import tqdm

SAMPLE_RATE = 16000
S_TO_MS = 1000


def parse_args(argv):
    parser = argparse.ArgumentParser(description='Prepare a dataset.')
    parser.add_argument('--file_cpc_split', type=str,
                        help='File to cpc splits')
    parser.add_argument('--transcription_file', type=str,
                        help='Path to the transcriptions per split')
    parser.add_argument('--snr_labels_file', type=str,
                        help='Path to the snr gold labels')
    parser.add_argument('--path_pred', type=str,
                        help='Path to the predictions of our model')
    parser.add_argument('--output_file', type=str,
                        help='Path to the output file')
    args = parser.parse_args(argv)
    return args


# add gold and predicted SNR values to the transcription file (.lst file)
# for wav2letter
def add_snr_to_transcriptions(
        file_cpc_split, transcription_file, path_pred, output_file):

    with open(transcription_file, 'r') as f:
        transcriptions = f.readlines()

    # load predicted SNR values
    i = 1
    pred_snr = torch.load(os.path.join(path_pred, f'snr/seq_{0}_pred.pt'))
    while os.path.exists(os.path.join(path_pred, f'snr/seq_{i}_pred.pt')):
        pred_snr = torch.cat(
            (pred_snr,
             torch.load(
                 os.path.join(
                     path_pred,
                     f'snr/seq_{i}_pred.pt'))))
        i += 1

    cpc_split = pd.read_csv(file_cpc_split, sep=' ')
    cpc_split.columns = ['filename', 'start', 'end', 'snr_gold', 'reverb_gold']
    cpc_split['snr_pred'] = pd.Series(pred_snr.cpu())
    cpc_split['start'] = cpc_split['start'] / (SAMPLE_RATE)
    cpc_split['end'] = cpc_split['end'] / (SAMPLE_RATE)
    cpc_split = cpc_split.sort_values(by=['filename', 'start'])

    offset = 0  # offset in the timestamps given by the cpc split file
    filename_prev = ''
    with open(output_file, 'w') as f:
        for line in tqdm(transcriptions):
            filename = line.split()[0]
            duration = line.split()[2]

            if filename.split(
                    '_')[-1] == '0':   # first segment of an audio file
                offset = cpc_split[cpc_split.filename == '_'.join(
                    filename.split('_')[:-1])].start.min()

            # start of the segment in the audio file
            start_seq = float(duration) * int(filename.split('_')[-1]) + offset
            # end of the segment in the audio file
            end_seq = float(duration) * \
                (int(filename.split('_')[-1]) + 1) + offset
            snr_gold = round(cpc_split[(cpc_split.filename == '_'.join(filename.split('_')[
                             :-1])) & (cpc_split.start >= start_seq) & (cpc_split.end <= end_seq)].snr_gold.mean(), 1)
            snr_pred = round(cpc_split[(cpc_split.filename == '_'.join(filename.split('_')[
                             :-1])) & (cpc_split.start >= start_seq) & (cpc_split.end <= end_seq)].snr_pred.mean(), 1)
            f.write(f'{filename} {duration} {snr_gold} {snr_pred}\n')


def main(argv):
    args = parse_args(argv)
    add_snr_to_transcriptions(
        args.file_cpc_split,
        args.transcription_file,
        args.path_pred,
        args.output_file)


if __name__ == "__main__":
    # execute only if run as a script
    args = sys.argv[1:]
    main(args)
