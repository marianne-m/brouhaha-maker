# import os
from tqdm import tqdm
from pathlib import Path
from cpc_dataset_maker.datasets.dataset import (
    Dataset,
    save_int_sequences,
)
from typing import List, Optional, Union
from cpc_dataset_maker.vad_pyannote.rttm_data import build_rttm_file_from_phone_labels

# create a file with the phone labels of an audio sequence
def load_phone_label_from_buckeye(path_buckeye_annot: Union[Path, str]) -> List[int]:
    with open(path_buckeye_annot) as f:
        content = [x.strip() for x in f.readlines() if len(x.strip()) > 0]
    start_index = next(i for i, x in enumerate(content) if x == "#") + 1
    word_times = []
    for item in content[
        start_index:
    ]:  # the first 9 lines of buckeye transcription files are general info
        if not item.startswith(";"):
            word_times += [item.split()[:3]]

    start = 0
    end = 1
    phone_labels = [0] * int(float(word_times[len(word_times) - 1][0]) * 100)
    # we consider that the timestamps given in transcription files
    # correspond to the end of each annotation
    while end < len(word_times):
        if word_times[start][2].startswith("<") or word_times[start][2].startswith(
            "{"
        ):  # end of non-speech region = start of a speech region
            start += 1
            end = start + 1
        # end of speech region = start of a non-speech region
        elif word_times[end][2].startswith("<") or word_times[end][2].startswith("{"):
            if not (
                word_times[start][2].startswith("<")
                or word_times[start][2].startswith("{")
            ):
                # phone labels have a time step of 10ms: 1s gives 100 label
                # points
                phone_labels[
                    int(float(word_times[start - 1][0]) * 100) : int(
                        float(word_times[end - 1][0]) * 100
                    )
                ] = [1] * (
                    int(float(word_times[end - 1][0]) * 100)
                    - int(float(word_times[start - 1][0]) * 100)
                )
            start = end
            end = start + 1
        else:
            end += 1

    return phone_labels


class Buckeye(Dataset):
    def __init__(
        self,
        root: Union[Path, str],
    ):
        Dataset.__init__(
            self,
            root=root,
            dataset_name="buckeye",
        )
        print(f"Working with Buckeye")

    def build_from_root_dir(
        self, path_root_buckeye: Union[Path, str], **kwargs
    ) -> None:
        self.resample(path_root_buckeye)
        self.create_phone_labels(path_root_buckeye)
        self.create_rttm(path_root_buckeye)

    # create a file with the phone labels of audio sequences
    def create_phone_labels(self, path_dir_labels: Union[Path, str]) -> None:
        files = list(Path(path_dir_labels).glob("**/*.words"))
        print(f"{len(files)} files found")
        phone_labels = {
            x.stem: load_phone_label_from_buckeye(path_dir_labels / x) for x in files
        }
        save_int_sequences(phone_labels, self.path_phone_labels)
        print(f"Phone labels saved at {self.path_phone_labels}")

    # create rttm files for all audio files
    def create_rttm(self, path_dir_labels: Union[Path, str]):
        files = list(Path(path_dir_labels).glob("**/*.words"))
        print(f"{len(files)} files found")
        self.path_rttm.mkdir(exist_ok=True)

        for rel_path in tqdm(files):
            phone_intervals = load_phone_label_from_buckeye(path_dir_labels / rel_path)
            path_rttm_out = (self.path_rttm / rel_path.name).with_suffix(".rttm")
            build_rttm_file_from_phone_labels(phone_intervals, path_rttm_out)
        print(f"RTTM files saved at {self.path_rttm}")

    # align audio transcriptions with an audio extended with silences
    def align_transcription_silence_file(
        self, transcription_file, rttm_init_file, rttm_silence_file, align_file
    ):
        with open(transcription_file, "r") as f:
            transcription = f.readlines()
        with open(rttm_silence_file, "r") as f:
            rttm_silence = f.readlines()
        with open(rttm_init_file, "r") as f:
            rttm_init = f.readlines()

        with open(align_file, "w") as f_align:
            idx_rttm = 0
            w_start = 0
            for line in transcription[9:]:
                if not (
                    line.split()[2].startswith("<") or line.split()[2].startswith("{")
                ):  # keep speech regions only
                    # look for the right segment in rttm_init
                    while idx_rttm < len(rttm_init) - 1 and w_start > float(
                        rttm_init[idx_rttm].split()[3]
                    ) + float(rttm_init[idx_rttm].split()[4]):
                        idx_rttm += 1

                    offset = float(
                        rttm_silence[
                            idx_rttm
                        ].split()[  # offset due to the extended silences
                            3
                        ]
                    ) - float(rttm_init[idx_rttm].split()[3])
                    w_end = float(line.split()[0])
                    line_align = f"{w_start + offset} {w_end + offset} {line.split()[2].replace(';','')}\n"
                    f_align.write(line_align)
                w_start = float(line.split()[0])

    # align audio transcriptions with audio extended with silences

    def align_transcription_silence(self):
        transcription_files = list(Path(self.path_transcription).glob("**/*.words"))
        rttm_silence_files = list(
            Path(self.path_rttm).glob("**/*.rttm")
        )  # rttm files with extended silences
        print(f"{len(transcription_files)} files found")

        for rttm_silence_file in tqdm(rttm_silence_files):
            filename = (
                os.path.splitext(str(rttm_silence_file))[0]
                .replace(self.path_rttm + "/", "")
                .replace(self.path_rttm, "")
            )

            rttm_init_file = os.path.join(
                os.path.dirname(self.path_rttm),  # initial rttm file
                "rttm_clean",
                filename + ".rttm",
            )
            transcription_file = os.path.join(
                self.path_transcription, filename + ".words"
            )  # initial transcription file (.words)
            align_file = os.path.join(
                os.path.dirname(self.path_rttm),
                "transcription_silence_smooth",
                filename + ".txt",
            )  # output aligned file
            os.makedirs(os.path.dirname(align_file), exist_ok=True)
            self.align_transcription_silence_file(
                transcription_file, rttm_init_file, rttm_silence_file, align_file
            )

        print(
            f"Transcription files saved at {os.path.join(os.path.dirname(self.path_rttm), 'transcription_silence_smooth')}"
        )
