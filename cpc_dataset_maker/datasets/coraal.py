import os
import csv
import glob
import wget
import shutil
from tqdm import tqdm
from pathlib import Path
import pandas as pd
from cpc_dataset_maker.datasets.dataset import (
    Dataset,
    save_int_sequences,
)
from typing import Dict, List, Optional, Set, Union
from cpc_dataset_maker.vad_pyannote.rttm_data import (
    build_rttm_file_from_phone_labels,
    speech_activities_to_int_sequence,
)


TRANSCRIPTS = [
    "http://lingtools.uoregon.edu/coraal/dca/2018.10.06/DCA_elanfiles_2018.10.06.tar.gz",
    "http://lingtools.uoregon.edu/coraal/dca/2018.10.06/DCA_textfiles_2018.10.06.tar.gz",
    "http://lingtools.uoregon.edu/coraal/dca/2018.10.06/DCA_textgrids_2018.10.06.tar.gz"
]
METADATA = "http://lingtools.uoregon.edu/coraal/dca/2018.10.06/DCA_metadata_2018.10.06.txt"

# create a file with the phone labels of an audio sequence
def load_phone_labels_from_coraal(
    path_coraal_annot: Union[str, Path], threshold_silence: float = 0.5
) -> List[int]:
    with open(path_coraal_annot, "r") as f_:
        reader = csv.DictReader(f_, delimiter="\t")
        last_start, last_end = None, None
        speech_activities = []
        for row in reader:
            start, end = float(row["StTime"]), float(row["EnTime"])
            if last_start is None:
                last_start, last_end = float(start), float(end)
                continue
            elif start > last_end + threshold_silence:
                speech_activities += [(last_start, last_end)]
                last_start, last_end = float(start), float(end)
            else:
                last_end = float(end)

    if last_start is not None:
        speech_activities += [(last_start, last_end)]

    return speech_activities_to_int_sequence(speech_activities, last_end)


class CORAAL(Dataset):
    def __init__(
        self,
        root: Union[Path, str],
    ):
        Dataset.__init__(
            self,
            root=root,
            dataset_name="coraal",
        )
        print(f"Working with CORAAL")

    def build_from_root_dir(self, path_root_coraal: Union[Path, str], extension: str, **kwargs) -> None:
        self.resample(path_root_coraal, original_extension=extension)
        self.create_phone_labels(path_root_coraal)
        self.create_rttm(path_root_coraal)

    # create a file with the phone labels of audio sequences
    def create_phone_labels(self, path_coraal_annot: Union[Path, str]) -> None:
        annot_files = [
            x for x in Path(path_coraal_annot).glob("**/*.txt")
        ]
        print(f"{len(annot_files)} files found")
        phone_labels = {
            x.stem: load_phone_labels_from_coraal(path_coraal_annot / x)
            for x in annot_files
        }
        save_int_sequences(phone_labels, self.path_phone_labels)
        print(f"Phone labels saved at {self.path_phone_labels}")

    # create rttm files for all audio files
    def create_rttm(self, path_coraal_annot: Union[Path, str]):
        annot_files = [
            x for x in Path(path_coraal_annot).glob("**/*.txt")
        ]
        self.path_rttm.mkdir(exist_ok=True)
        for path_annot in annot_files:
            phone_seq = load_phone_labels_from_coraal(path_coraal_annot / path_annot)
            build_rttm_file_from_phone_labels(
                phone_seq, (self.path_rttm / path_annot.name.replace(".txt", "_vad.rttm"))
            )
        print(f"RTTM files saved at {self.path_rttm}")

    # align audio transcriptions with an audio extended with silences

    def align_transcription_silence_file(
        self, transcription_file, rttm_init_file, rttm_silence_file, align_file
    ):
        transcription = pd.read_csv(transcription_file, sep="\t")
        transcription = transcription.sort_values(by="StTime")
        with open(rttm_silence_file, "r") as f:
            rttm_silence = f.readlines()
        with open(rttm_init_file, "r") as f:
            rttm_init = f.readlines()

        with open(align_file, "w") as f_align:
            idx_rttm = 0
            for i in range(len(transcription)):
                if not transcription.Content[i].startswith(
                    "(pause"
                ):  # keep speech regions only
                    w_start = float(transcription.StTime[i])
                    w_end = float(transcription.EnTime[i])
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
                    line_align = f"{w_start + offset} {w_end + offset} {transcription.Content[i]}\n"
                    f_align.write(line_align)

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
            transcription_file = os.path.join(  # initial transcription file (.txt)
                self.path_transcription, filename + ".txt"
            )
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
    
    def download_datasets(self, downloaded_path):
        print(f"Downloading CORAAL DCA dataset into {downloaded_path}...")
        coraal = Path(downloaded_path)
        coraal.mkdir(parents=True, exist_ok=True)

        parts = [str(num).zfill(2) for num in range(1, 11)]
        for part in tqdm(parts, total=len(parts)):
            url = f"http://lingtools.uoregon.edu/coraal/dca/2018.10.06/DCA_audio_part{part}_2018.10.06.tar.gz"
            wget.download(url)
            archive = Path(url).name
            extract_dir = str(coraal / archive).replace(".tar.gz", "")
            shutil.unpack_archive(archive, extract_dir)
            os.remove(archive)

        for file in TRANSCRIPTS:
            wget.download(file)
            archive = Path(file).name
            extract_dir = coraal / (archive).replace(".tar.gz", "")
            shutil.unpack_archive(archive, extract_dir)
            os.remove(archive)
        
        wget.download(METADATA)
        hidden_files = glob.glob(f"{coraal}/**/.*")
        for file in hidden_files:
            os.remove(file)

        print("\nDone.")