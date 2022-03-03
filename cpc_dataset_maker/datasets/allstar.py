import os
from tqdm import tqdm
from pathlib import Path
from cpc_dataset_maker.datasets.dataset import (
    Dataset,
    save_int_sequences,
    read_speech_sequence_from_text_grid,
)
from cpc_dataset_maker.vad_pyannote.rttm_data import build_rttm_file_from_phone_labels
from typing import List, Optional, Union


class ALLSTAR(Dataset):
    def __init__(
        self,
        root: Union[Path, str],
        threshold_silence: float = 0.5,
    ):
        Dataset.__init__(
            self,
            root=root,
            dataset_name="allsstar",
        )
        self.threshold_silence = threshold_silence
        print(f"Working with ALLSTAR")
        print(f"Minimum silences of {threshold_silence}s")

    def build_from_root_dir(
        self, path_root_allsstar: Union[Path, str], **kwargs
    ) -> None:
        # self.resample(path_root_allsstar, '.wav')
        print("Building the phone labels")
        self.create_phone_labels(path_root_allsstar)
        self.create_rttm(path_root_allsstar)

    # create a file with the phone labels of audio sequences
    def create_phone_labels(self, path_labels: Union[Path, str]):
        files = list(Path(path_labels).glob("**/*.TextGrid"))
        print(f"{len(files)} files found")
        phone_labels = {
            x.stem: read_speech_sequence_from_text_grid(
                path_labels / x, self.threshold_silence
            )
            for x in files
        }
        save_int_sequences(phone_labels, self.path_phone_labels)
        print(f"Phone labels saved at {self.path_phone_labels}")

    # create rttm files for all audio files
    def create_rttm(self, path_labels: Union[Path, str]) -> None:
        files = list(Path(path_labels).glob("**/*.TextGrid"))
        self.path_rttm.mkdir(exist_ok=True)
        print(f"{len(files)} files found")

        for rel_path in tqdm(files):
            phone_intervals = read_speech_sequence_from_text_grid(
                path_labels / rel_path, self.threshold_silence
            )
            build_rttm_file_from_phone_labels(
                phone_intervals, (self.path_rttm / rel_path.name).with_suffix(".rttm")
            )
        print(f"RTTM files saved at {self.path_rttm}")

    # align audio transcriptions with an audio extended with silences
    def align_transcription_silence_file(
        self, transcription_file, rttm_init_file, rttm_silence_file, align_file
    ):
        transcription = tgt.read_textgrid(transcription_file)
        with open(rttm_silence_file, "r") as f:
            rttm_silence = f.readlines()
        with open(rttm_init_file, "r") as f:
            rttm_init = f.readlines()

        with open(align_file, "w") as f_align:
            if transcription.has_tier("Speaker - word") or transcription.has_tier(
                "utt - words"
            ):
                try:
                    intervals = transcription.get_tier_by_name(
                        "Speaker - word"
                    ).intervals
                except BaseException:
                    intervals = transcription.get_tier_by_name("utt - words").intervals

                idx_rttm = 0
                for interval in intervals:
                    if interval.text != "sp":  # keep speech regions only
                        w_start = float(interval.start_time)
                        w_end = float(interval.end_time)
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
                        line_align = (
                            f"{w_start + offset} {w_end + offset} {interval.text}\n"
                        )
                        f_align.write(line_align)
            else:
                print(f"Could not extract word timeline for {transcription_file}")
                return

    # align audio transcriptions with audio extended with silences

    def align_transcription_silence(self):
        transcription_files = list(Path(self.path_transcription).glob("**/*.words"))
        rttm_silence_files = list(Path(self.path_rttm).glob("**/*.rttm"))
        print(f"{len(transcription_files)} files found")

        for rttm_silence_file in tqdm(rttm_silence_files):
            filename = (
                os.path.splitext(str(rttm_silence_file))[0]
                .replace(self.path_rttm + "/", "")
                .replace(self.path_rttm, "")
            )  # rttm files with extended silences

            rttm_init_file = os.path.join(
                os.path.dirname(self.path_rttm), "rttm", filename + ".rttm"
            )  # initial rttm file
            transcription_file = os.path.join(
                self.path_transcription, filename + ".TextGrid"
            )  # initial transcription file (.TextGrid)
            align_file = os.path.join(
                os.path.dirname(self.path_rttm),
                "transcription_silence_smoth",
                filename + ".txt",
            )  # output aligned file
            os.makedirs(os.path.dirname(align_file), exist_ok=True)
            self.align_transcription_silence_file(
                transcription_file, rttm_init_file, rttm_silence_file, align_file
            )

        print(
            f"Transcription files saved at {os.path.join(os.path.dirname(self.path_rttm), 'transcription_silence_smooth')}"
        )
