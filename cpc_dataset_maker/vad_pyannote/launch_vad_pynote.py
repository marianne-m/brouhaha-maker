from pathlib import Path
import argparse
import torch
import json
import sys

# from pyannote.audio.utils.signal import Binarize
# from pyannote.audio.features import Pretrained

from cpc.dataset import find_seqs_relative
from progressbar import ProgressBar

from seq_name_dataset import SeqNameDataset
from vad_feeder import VADFeeder
from vad_pyx.dtw import build_vad_intervals
from typing import Any, Union
from cpc_dataset_maker.vad_pyannote.rttm_data import save_speech_activities_to_rttm


def run_model(data_loader, model: torch.nn.Module, vad_feeder: VADFeeder) -> None:

    bar = ProgressBar(maxval=len(data_loader))
    bar.start()
    i = 0

    with torch.no_grad():
        for batch_data, seq_index, chunk_index in data_loader:

            bar.update(i)
            batch_data = batch_data.cuda(non_blocking=True)
            out = model(batch_data)
            out = torch.exp(out)
            vad_feeder.feed_seq_data(seq_index, chunk_index, out.cpu())
            i += 1

    bar.finish()


def save_vad_data(
    vad_feeder: VADFeeder,
    path_out: Union[Path, str],
    cfg_bin: Any,
    format: str = "json",
    squash_output: bool = True,
):

    path_out = Path(path_out)
    path_out.mkdir(exist_ok=True)
    bar = ProgressBar(maxval=vad_feeder.n_seqs)
    bar.start()

    for index in range(vad_feeder.n_seqs):
        bar.update(index)
        vad_vector = vad_feeder.get_vad(index)
        vad_intervals = build_vad_intervals(
            vad_vector,
            cfg_bin.time_chunk,
            cfg_bin.onset,
            cfg_bin.offset,
            cfg_bin.offset_time_chunk,
            cfg_bin.pad_start,
            cfg_bin.pad_end,
            cfg_bin.min_size_sil,
            cfg_bin.min_size_voice,
        )

        # Build the output file
        seq_name = Path(vad_feeder.get_seq_name(index))
        if squash_output:
            full_path_out = path_out / f"{seq_name.stem}_vad.{format}"
        else:
            full_path_out = path_out / seq_name.parent / f"{seq_name.stem}_vad.{format}"
            full_path_out.parent.mkdir(exist_ok=True, parents=True)

        if format == "json":
            with open(str(full_path_out), "w") as file:
                json.dump(vad_intervals, file, indent=2)
        elif format == "rttm":
            save_speech_activities_to_rttm(vad_intervals, full_path_out)

    bar.finish()


def parse_args(argv):

    parser = argparse.ArgumentParser(
        description="VAD from pyannote backend. "
        "This python script will launch the pyannote model on all available GPUs"
    )

    # Default arguments:
    parser.add_argument("path_db", help="Root directory of the dataset")
    parser.add_argument("--file_extension", type=str, default=".wav")
    parser.add_argument(
        "-o",
        "--path_out",
        type=str,
        required=True,
        help="Output directory where the vad files should be saved",
    )
    parser.add_argument(
        "--out_format",
        type=str,
        default="rttm",
        choices=["json", "rttm"],
        help="Output format of the vad",
    )
    parser.add_argument("--size_batch", type=int, default="64")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Activate to load only a small part of the dataset",
    )
    parser.add_argument(
        "--keep-db-structure",
        action="store_true",
        help="Reprooduce the file hierarchy of the input dataset when "
        "writing the vad files.",
    )

    cfg_bin = parser.add_argument_group("cfg_bin", "Configuration of the binarizer")
    cfg_bin.add_argument(
        "--offset",
        type=float,
        default=0.6133102927330462,
        help="Score threshold to switch off the voice activity.",
    )
    cfg_bin.add_argument(
        "--onset",
        type=float,
        default=0.6133102927330462,
        help="Score threshold to switch on the voice activity.",
    )
    cfg_bin.add_argument(
        "--min_size_sil",
        type=float,
        default=0.3,
        help="Minimal size of a silent segment",
    )
    cfg_bin.add_argument(
        "--min_size_voice",
        type=float,
        default=1.0,
        help="Minimal size of a voice segment",
    )
    cfg_bin.add_argument(
        "--pad_start",
        type=float,
        default=0,
        help="Left padding (in seconds) applied to the final audio segments",
    )
    cfg_bin.add_argument(
        "--pad_end",
        type=float,
        default=0,
        help="Right padding (in seconds) applied to the final audio segments",
    )
    cfg_bin.add_argument(
        "--duration",
        type=float,
        default=None,
        help="Duration of each segment in the batch at inference",
    )
    return parser.parse_args(argv)


def main(args):

    seq_list = find_seqs_relative(args.path_db, extension=args.file_extension)

    if args.debug:
        seq_list = seq_list[:10]

    # The model
    print("Loading the model")
    base_model = torch.hub.load("pyannote/pyannote-audio", "sad_ami")
    sad_model = base_model.model_
    sad_model.eval().cuda()
    sad_model = torch.nn.DataParallel(
        sad_model, device_ids=range(torch.cuda.device_count())
    )

    # Resolution data
    segment_duration = base_model.duration if args.duration is None else args.duration
    sample_rate = base_model.feature_extraction_.sample_rate
    size_frame = int(segment_duration * sample_rate)

    vad_resolution = base_model.get_resolution()
    vad_step_size = vad_resolution.step
    vad_window_size = vad_resolution.duration

    args.offset_time_chunk = 0  # vad_window_size / 2
    args.time_chunk = vad_step_size

    n_frames_vad = int(vad_step_size * sample_rate)
    size_vad_output = (
        int((size_frame - vad_window_size * sample_rate) / n_frames_vad) + 1
    )

    # Dataset
    print("Loading the dataset")
    size_cut = 100
    n_cut = len(seq_list) // size_cut + 1
    for p in range(n_cut):
        print(f"Group {p+1} out of {n_cut}")
        loc_interval_seqs = seq_list[p * size_cut : p * size_cut + size_cut]
        dataset = SeqNameDataset(
            args.path_db, loc_interval_seqs, size_frame, args.size_batch
        )

        vad_feeder = VADFeeder(
            loc_interval_seqs, args.path_db, size_frame, size_vad_output
        )

        print("Starting the VAD computation")
        run_model(dataset, sad_model, vad_feeder)

        print(f"Saving the VAD intervals at {args.path_out}")
        save_vad_data(
            vad_feeder, args.path_out, args, args.out_format, not args.keep_db_structure
        )


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn", force=True)
    args = sys.argv[1:]
    main(parse_args(args))
