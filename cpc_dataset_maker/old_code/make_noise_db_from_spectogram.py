import torchaudio
import numpy as np
import torch
import multiprocessing
import tqdm
import sys
import argparse
from random import shuffle
from pathlib import Path
from cpc.dataset import findAllSeqs
from progressbar import ProgressBar


class SpectogramDBMaker:
    def __init__(self, n_bins, window=None, normalized=True, hop_length=None):
        self.n_bins = n_bins
        self.window = window
        self.normalized = normalized
        self.hop_length = n_bins // 2 if hop_length is None else n_bins

    def get_spectogram(self, path_seq):

        out_data = np.zeros(self.n_bins)
        seq = torchaudio.load(path_seq)[0].mean(dim=0)
        if seq.size(0) < self.hop_length:
            return None
        return torch.stft(
            seq,
            self.n_bins,
            win_length=self.n_bins,
            window=self.window,
            normalized=self.normalized,
            hop_length=self.hop_length,
        )


def get_spectogram(data):

    path_seq, n_bins = data
    out_data = np.zeros(n_bins)
    seq = torchaudio.load(path_seq)[0].mean(dim=0)
    return torch.stft(
        seq,
        n_bins,
        win_length=n_bins,
        window=torch.hann_window(n_bins),
        normalized=True,
        hop_length=n_bins // 2,
    )


def get_module(x):

    assert x.size(-1) == 2
    return torch.sqrt((x ** 2).sum(dim=2))


def get_avg_modulus_spectogram(
    path_db, seq_list, n_bins, pool, n_processes=4, hop_length=None, window=None
):

    spectogram_amker = SpectogramDBMaker(
        n_bins, window=window, normalized=True, hop_length=hop_length
    )

    path_db = Path(path_db)
    to_process = [str(path_db / x) for x in seq_list]
    out = torch.zeros(n_bins // 2 + 1)
    N = 0

    for fft_ in tqdm.tqdm(
        pool.imap_unordered(spectogram_amker.get_spectogram, to_process),
        total=len(to_process),
    ):
        if fft_ is not None:
            p = get_module(fft_)
            out += p.mean(dim=1)
            N += 1

    return out / N


def build_noise_db_from_spectrum(
    dir_out, n_sequences, ref_spectrum, n_bins, size_out=30, sr=16000, window=None
):

    dir_out = Path(dir_out)
    dir_out.mkdir(exist_ok=True)
    size_seq = size_out * sr
    multiplier = ref_spectrum.view(-1, 1, 1)
    for n in range(n_sequences):
        data = torch.randn(size_seq)
        data = torch.stft(
            data,
            n_bins,
            win_length=n_bins,
            window=window,
            normalized=True,
            hop_length=n_bins // 2,
        )

        data = data * multiplier
        data = torchaudio.functional.istft(
            data,
            n_bins,
            win_length=n_bins,
            window=window,
            normalized=True,
            hop_length=n_bins // 2,
        )
        data /= data.abs().max()

        path_out = str(dir_out / f"{n}_.flac")
        torchaudio.save(path_out, torch.clamp(data, min=-1, max=1), sr)


def apply_amplifier_to_noise_db(
    dir_in, seq_list, amplifier, dir_out, n_bins, window=None
):

    dir_out = Path(dir_out)
    dir_out.mkdir(exist_ok=True)

    dir_in = Path(dir_in)
    amplifier = amplifier.view(-1, 1, 1)

    bar = ProgressBar(maxval=len(seq_list))
    bar.start()
    for index, seq_name in enumerate(seq_list):
        bar.update(index)
        try:
            data, sr = torchaudio.load(str(dir_in / seq_name))
            data = data.mean(dim=0)

            data = torch.stft(
                data,
                n_bins,
                win_length=n_bins,
                window=window,
                normalized=True,
                hop_length=n_bins // 4,
            )
            data = data * amplifier
            data = torchaudio.functional.istft(
                data,
                n_bins,
                win_length=n_bins,
                window=window,
                normalized=True,
                hop_length=n_bins // 4,
            )
            data /= data.abs().max()

            path_out = dir_out / seq_name
            path_out.parent.mkdir(exist_ok=True, parents=True)
            torchaudio.save(str(path_out), torch.clamp(data, min=-1, max=1), sr)
        except RuntimeError:
            continue

    bar.finish()


def build_band_pass_spectrum(min_freq, max_freq, samplig_rate, fft_window_size):

    fft_res = samplig_rate / fft_window_size
    out = torch.zeros(fft_window_size // 2 + 1)
    start_index = int(min_freq / fft_res)
    end_index = int(max_freq / fft_res)
    out[start_index:end_index] = 1
    return out


def parse_args(argv):

    parser = argparse.ArgumentParser(description="Noise builder")
    subparsers = parser.add_subparsers(dest="command")

    parser_sp = subparsers.add_parser("make_noise_from_spectrum")
    parser_sp.add_argument("path_db")
    parser_sp.add_argument("--file_extension", type=str, default=".wav")
    parser_sp.add_argument("-o", "--output", type=str, default="coin")

    parser_sm = subparsers.add_parser("spectrum_multiplier")
    parser_sm.add_argument("path_ref")
    parser_sm.add_argument("path_in")
    parser_sm.add_argument("--file_extension_ref", type=str, default=".flac")
    parser_sm.add_argument("--file_extension_in", type=str, default=".wav")
    parser_sm.add_argument("-o", "--output", type=str, default="coin")
    parser_sm.add_argument("--fft_window_size", type=int, default=512)
    parser_sm.add_argument("--multiplier", type=float, default=1)

    parser_bp = subparsers.add_parser("band_pass")
    parser_bp.add_argument("-o", "--output", type=str, default="coin")
    parser_bp.add_argument("--path_db", type=str, default=None)
    parser_bp.add_argument("--file_extension", type=str, default=".wav")
    parser_bp.add_argument("--min_freq", type=int, default=0)
    parser_bp.add_argument("--max_freq", type=int, default=80)
    parser_bp.add_argument("--fft_window_size", type=int, default=512)
    parser_bp.add_argument("--sampling_rate", type=int, default=16000)
    parser_bp.add_argument("--debug", action="store_true")

    return parser.parse_args(argv)


def make_noise_from_spectrum(args):

    seq_list = [
        x[1] for x in findAllSeqs(args.path_db, extension=args.file_extension)[0]
    ]
    ref_spec = get_avg_modulus_spectogram(args.path_db, seq_list, 512)
    build_noise_db_from_spectrum(
        args.output, 200, ref_spec, 512, window=torch.hann_window(n_bins)
    )


def band_pass(args):

    ref_spec = build_band_pass_spectrum(
        args.min_freq, args.max_freq, args.sampling_rate, args.fft_window_size
    )
    if args.path_db is not None:
        seq_list = [
            x[1]
            for x in findAllSeqs(
                args.path_db, extension=args.file_extension, loadCache=False
            )[0]
        ]
        if args.debug:
            seq_list = seq_list[:3]

        print(len(seq_list))

        apply_amplifier_to_noise_db(
            args.path_db,
            seq_list,
            ref_spec,
            args.output,
            args.fft_window_size,
            window=torch.hann_window(args.fft_window_size),
        )
    else:
        build_noise_db_from_spectrum(
            args.output,
            200,
            ref_spec,
            args.fft_window_size,
            window=torch.hann_window(args.fft_window_size),
        )


def spectrum_multiplier(args):

    window = torch.hann_window(args.fft_window_size)
    with multiprocessing.Pool(processes=4) as pool:
        seq_list_ref = [
            x[1]
            for x in findAllSeqs(args.path_ref, extension=args.file_extension_ref)[0]
        ]
        db_specs = get_avg_modulus_spectogram(
            args.path_ref,
            seq_list_ref,
            args.fft_window_size,
            pool,
            window=window,
            hop_length=args.fft_window_size // 4,
        )

        seq_list_in = [
            x[1] for x in findAllSeqs(args.path_in, extension=args.file_extension_in)[0]
        ]
        in_specs = get_avg_modulus_spectogram(
            args.path_in,
            seq_list_in,
            args.fft_window_size,
            pool,
            window=window,
            hop_length=args.fft_window_size // 4,
        )

    in_specs = in_specs + 1.0 * (torch.abs(in_specs) < 1e-8)
    multiplier = args.multiplier * torch.abs(db_specs / (in_specs))
    print(multiplier.min(), multiplier.max())
    apply_amplifier_to_noise_db(
        args.path_in,
        seq_list_in,
        multiplier,
        args.output,
        args.fft_window_size,
        window=window,
    )


if __name__ == "__main__":
    argv = sys.argv[1:]
    args = parse_args(argv)

    if args.command == "make_noise_from_spectrum":
        make_noise_from_spectrum(args)
    if args.command == "spectrum_multiplier":
        spectrum_multiplier(args)
    else:
        band_pass(args)
