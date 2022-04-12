from pathlib import Path
import argparse
from cpc_dataset_maker.transforms.segmentation import Segmentation
import torch
from cpc_dataset_maker.datasets import get_dataset_builder, AVAILABLE_DATASETS, Dataset
from cpc_dataset_maker.datasets.transformed_dataset import (
    TransformDataset,
    update_audio_labels,
)
from cpc_dataset_maker.transforms.transform import CombinedTransform
from cpc_dataset_maker.transforms import (
    get_transform,
    PROBA_NO_REVERB,
    AVAILABLE_TRANSFORMS,
)
from cpc_dataset_maker.transforms.labels import SPEECH_ACTIVITY_LABEL


def init(args):

    base_db = get_dataset_builder(args.dataset_name)(args.root_db)

    if args.root_in is not None:
        print("Building the dataset")
        base_db.build_from_root_dir(args.root_in, args.file_extension)


def transform(args):

    base_db = get_dataset_builder(args.dataset_name)(args.root_db)

    out_db_dir = Path(args.output_dir)
    out_db = TransformDataset(out_db_dir, f"{args.dataset_name}_{args.name}")

    # For now, load all labels
    labels = out_db.init_audio_labels(base_db.get_all_files())
    update_audio_labels(labels, base_db.load_voice_activity(), SPEECH_ACTIVITY_LABEL)

    transform_list = [
        get_transform(transform_name, **vars(args))
        for transform_name in args.transforms
    ]
    full_transform = CombinedTransform(transform_list)

    if args.debug:
        labels = labels[:10]
    out_db.build(labels, full_transform, n_process=1)


def segment(args):

    base_db = get_dataset_builder(args.dataset_name)(args.root_db)
    out_db_dir = Path(args.output_dir)
    out_db = TransformDataset(out_db_dir, f"{args.dataset_name}_segmentation")

    segmentation = Segmentation(args.target_size)
    # For now, load all labels
    labels = out_db.init_audio_labels(base_db.get_all_files())
    update_audio_labels(labels, base_db.load_voice_activity(), SPEECH_ACTIVITY_LABEL)

    if args.debug:
        labels = labels[:1]
    out_db.build(labels, segmentation, n_process=1)


def update_base_parser(parser):
    parser.add_argument(
        "dataset_name",
        type=str,
        help="Name of the dataset",
        choices=AVAILABLE_DATASETS,
    )
    parser.add_argument(
        "root_db", type=str, help="Path where the output dataset should be saved"
    )
    parser.add_argument("--debug", action="store_true")


def parse_args():

    parser = argparse.ArgumentParser("Build the datasets for the SNR / VAD prediction")
    subparsers = parser.add_subparsers(dest="command")
    parser_init = subparsers.add_parser("init")
    update_base_parser(parser_init)
    parser_init.add_argument(
        "--root-in",
        type=str,
        help="Path of the dataset as downloaded from source. "
        "Give this argument to build the SNR / VAD modified dataset",
        default=None,
    )
    parser_init.add_argument("--file_extension", type=str, default=".flac")

    parser_transform = subparsers.add_parser("transform")
    update_base_parser(parser_transform)
    parser_transform.add_argument("-o", "--output-dir", type=str, required=True)
    parser_transform.add_argument(
        "--name", type=str, default="16k_transformed", help="Name of the transformation"
    )
    parser_transform.add_argument(
        "--transforms",
        type=str,
        nargs="*",
        choices=AVAILABLE_TRANSFORMS,
        required=True,
    )
    group_extend_sil = parser_transform.add_argument_group(
        "Silence extension", description="Arguments for the silence extension."
    )
    group_extend_sil.add_argument(
        "--cossfade-sec",
        type=float,
        default=0.1,
        help="Lenght (in second) of the crossfading when extending the silences of a dataset",
    )
    group_extend_sil.add_argument(
        "--target-share-sil",
        type=float,
        default=0.2,
        help="Target silence / voice ratio in the silence extension",
    )
    group_extend_sil.add_argument(
        "--sil-mean-sec",
        type=float,
        default=2.0,
        help="Mean lenght (in sec) of the random silences added to the audio",
    )
    group_extend_sil.add_argument(
        "--expand-silence-only",
        action="store_true",
        help="If set to true, add silence only to non spech regions",
    )

    group_extend_reverb = parser_transform.add_argument_group(
        "Reverberation", description="Arguments for the reverberation"
    )
    group_extend_reverb.add_argument(
        "--dir-impulse-responses",
        type=str,
        help="Directory containing a set of impulse responses for the reverberation",
    )
    group_extend_reverb.add_argument(
        "--ext-impulse",
        type=str,
        help="File extension of the impulse response data",
        default=".flac",
    )
    group_extend_reverb.add_argument(
        "--tau", type=float, default=50.0, help="Tau value for the c50 measure in ms"
    )
    group_extend_reverb.add_argument(
        "--proba-no-reverb",
        type=float,
        default=PROBA_NO_REVERB,
        help="Probability to not add reverberation to a file",
    )

    group_extend_noise = parser_transform.add_argument_group(
        "Noise augmentation", description="Arguments for the noise augmentation"
    )
    group_extend_noise.add_argument(
        "--dir-noise",
        type=str,
        help="Directory of the noise dataset",
    )
    group_extend_noise.add_argument(
        "--ext-noise",
        type=str,
        help="Extension of the noise audio files",
        default=".flac",
    )
    group_extend_noise.add_argument(
        "--snr-min", type=float, help="Minimal value of the snr", default=0.01
    )
    group_extend_noise.add_argument(
        "--snr-max", type=float, help="Maximal value of the snr", default=30
    )

    parser_segment = subparsers.add_parser("segmentation")
    update_base_parser(parser_segment)
    parser_segment.add_argument("-o", "--output-dir", type=str, required=True)
    parser_segment.add_argument("-t", "--target-size", type=float, default=10.0)

    return parser.parse_args()


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn", force=True)
    args = parse_args()

    if args.command == "init":
        init(args)
    if args.command == "transform":
        transform(args)
    if args.command == "segmentation":
        segment(args)
