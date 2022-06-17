from genericpath import exists
import os
from random import shuffle
import shutil
import torchaudio
import librosa
import tgt
import math
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from unidecode import unidecode
from multiprocessing import Pool
from cpc.dataset import find_seqs_relative
from typing import Dict, List, Optional, Set, Tuple, Union
from cpc_dataset_maker.vad_pyannote.rttm_data import (
    build_rttm_file_from_phone_labels,
    load_speech_activities_from_rttm,
    save_speech_activities_to_rttm,
    speech_activities_to_int_sequence,
)
from cpc_dataset_maker.transforms.add_noise import save_detailed_snr_labels

CPC_SPLIT_LENGTH = 0.01
TIME_STEP_PYANNOTE = 0.0203125
AVAILABLE_LANGUAGES = {"en", "fr"}
SR_CPC = 16000


def load_sequence_file(path_data: Union[str, Path]) -> List[str]:
    with open(path_data, "r") as f_:
        return [x.strip() for x in f_]


def save_sequence_file(data: List[str], path_out: Union[str, Path]) -> None:
    with open(path_out, "w") as f_:
        for x in data:
            f_.write(x + "\n")


def load_int_sequences(path_phone_labels: Union[Path, str]) -> Dict[str, List[int]]:
    with open(path_phone_labels, "r") as f:
        data = f.readlines()
    phone_labels = {}
    for line in data:
        phone_labels[line.split()[0]] = [int(x) for x in line.split()[1:]]

    return phone_labels


def save_int_sequences(
    phone_labels: Dict[str, List[int]], path_out: Union[Path, str]
) -> None:
    with open(path_out, "w") as f_:
        for filename, sequence in phone_labels.items():
            f_.write(f"{filename} " + " ".join([str(x) for x in sequence]) + "\n")


def load_float_sequences(path_float_labels: Union[Path, str]) -> Dict[str, List[float]]:
    with open(path_float_labels, "r") as f:
        data = [x.strip() for x in f.readlines()]
    float_labels = {}
    for line in data:
        float_labels[line.split()[0]] = float(line.split()[1])
    return float_labels


def resample_file(data):
    pathIn, pathOut, new_sr = data
    if not os.path.isfile(pathIn):
        return

    data, sr = torchaudio.load(pathIn)
    sampler = torchaudio.transforms.Resample(
        orig_freq=sr, new_freq=new_sr, resampling_method="sinc_interpolation"
    )
    data = sampler(data)
    torchaudio.save(pathOut, data, new_sr)


# create a file with the phone labels of an audio sequence
def read_speech_sequence_from_text_grid(
    file: Union[Path, str], threshold_silence: float
) -> List[float]:

    textgrid = tgt.read_textgrid(file)

    if textgrid.has_tier("Speaker - word") or textgrid.has_tier("utt - words"):
        try:
            intervals = textgrid.get_tier_by_name("Speaker - word").intervals
        except BaseException:
            intervals = textgrid.get_tier_by_name("utt - words").intervals
        phone_labels = [0] * int(intervals[len(intervals) - 1].end_time * 100)

        end_time_prev = -1
        for interval in intervals:
            if interval.text != "sp":  # filter out non-speech region
                # phone labels have a time step of 10ms: 1s gives 100 label
                # points
                if (
                    end_time_prev != -1
                    and interval.start_time - end_time_prev < threshold_silence
                ):  # minimum silence length between two speech regions
                    phone_labels[
                        int(end_time_prev * 100) : int(interval.end_time * 100)
                    ] = [1] * (int(interval.end_time * 100) - int(end_time_prev * 100))
                else:
                    phone_labels[
                        int(interval.start_time * 100) : int(interval.end_time * 100)
                    ] = [1] * (
                        int(interval.end_time * 100) - int(interval.start_time * 100)
                    )
                end_time_prev = interval.end_time
        return phone_labels
    else:
        print("Could not extract word timeline for", file)
        return None


def load_tokens(path_tokens: Union[Path, str]) -> Set[str]:
    with open(path_tokens, "r") as f_tok:
        token_list = f_tok.readlines()
    token_list = [tok.replace("\n", "") for tok in token_list]
    token_list += [" "]
    return set(token_list)


def load_lexicon(path_lexicon: Union[Path, str]) -> Set[str]:
    lexicon_list = load_sequence_file(path_lexicon)
    lexicon_list = [lex.split()[0] for lex in lexicon_list]
    return set(lexicon_list)


def get_path_token_from_lang(lang: str) -> Path:

    if lang not in AVAILABLE_LANGUAGES:
        raise ValueError(
            f"Invalid lang code : {lang}.\n"
            f"Available languages are : {AVAILABLE_LANGUAGES}"
        )
    dir_tokens = Path(__file__).parent.parent / "data"
    return dir_tokens / f"{lang}_tokens.txt"


def get_path_lexicon_from_lang(lang: str) -> Path:
    if lang not in AVAILABLE_LANGUAGES:
        raise ValueError(
            f"Invalid lang code : {lang}.\n"
            f"Available languages are : {AVAILABLE_LANGUAGES}"
        )
    dir_tokens = Path(__file__).parent.parent / "data"
    return dir_tokens / f"{lang}_lexicon.txt"


# split an audio file into shorter segments
def split_audio_file(
    file: Union[Path, str], length: float, out_ext: Optional[str] = None
) -> None:
    audio, sr = torchaudio.load(file)
    seqs = audio.split(int(length * sr), dim=1)
    id = 0

    if out_ext is None:
        out_ext = Path(file).suffix

    for seq in seqs:
        # if the last segment is shorter than the split size, we skip it
        # (likewise CPC)
        if seq.shape[1] == length * sr:
            output_file = os.path.join(
                os.path.splitext(str(file))[0] + "_split",
                os.path.splitext(os.path.basename(file))[0] + f"_{id}" + out_ext,
            )
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            torchaudio.save(output_file, seq, sr)
            id += 1


# clean transcriptions: lowercase, remove accents, tokens and lexicon
# filters
def clean_transcription(text: str, language: str) -> str:
    token_set = load_tokens(get_path_token_from_lang(language))
    lexicon_set = load_lexicon(get_path_lexicon_from_lang(language))

    text_lowercase = text.lower()  # lowercase
    text_no_accent = unidecode(text_lowercase)  # remove accents
    text_filter_token = "".join(  # filter tokens
        [tok for tok in text_no_accent if tok in token_set]
    )
    text_filter_lexicon = " ".join(  # filter lexicon
        [lex for lex in text_filter_token.split() if lex in lexicon_set]
    )
    return text_filter_lexicon


# extract the transcription of an audio segment
def extract_transcription_seq(
    transcription_file: Union[Path, str], start: float, end: float, language: str
) -> str:
    with open(transcription_file, "r") as f:
        transcription = f.readlines()

    text = ""
    for line in transcription:
        w_start = float(line.split()[0])
        w_end = float(line.split()[1])
        if w_end >= start and w_start <= end:
            words = line.split()[2:]
            if (
                w_start < start or w_end > end
            ):  # when a span of words exceeds a segement, we
                offset_start = max(
                    (start - w_start)
                    / (  # split it proprotionately to the segment length
                        w_end - w_start
                    ),
                    0,
                )
                offset_end = min((end - w_start) / (w_end - w_start), 1)
                words = words[
                    math.floor(len(words) * offset_start) : math.ceil(
                        len(words) * offset_end
                    )
                ]
            text += " ".join(words) + " "
        elif w_start > end:
            break

    text_clean = clean_transcription(text, language)  # we clean transcription texts
    return text_clean


class Dataset:
    def __init__(
        self,
        root: Union[Path, str],
        dataset_name: str = "base",
        sr_db: int = SR_CPC,  # sample rate of audio files
        ext_db: str = ".flac",  # extension of audio files
        path_training_set: Optional[
            Union[str, Path]
        ] = None,  # path to the training set file
        path_test_set: Optional[Union[str, Path]] = None,  # path to the test set file
        path_data_split: Union[str, Path] = None,
        path_pred: Union[str, Path] = None,
        path_phone_labels: Union[str, Path] = None,
        path_snr_labels: Union[str, Path] = None,
        path_detailed_snr_labels: Union[str, Path] = None,
        path_reverb_labels: Union[str, Path] = None,
        ignore_cache: bool = False,  # ignore existing cache files
    ):
        self.root = Path(root)
        self.root.mkdir(exist_ok=True)

        self.ext_db = ext_db
        self.sr_db = sr_db
        self.dataset_name = dataset_name

        # path to the file with the data split made by CPC
        self.path_data_split = path_data_split
        self.path_pred = path_pred  # path to CPC predictions

        self.ignore_cache = ignore_cache

        if path_training_set is not None:
            self.use_training_set(path_training_set)
        if path_test_set is not None:
            self.use_test_set(path_test_set)
        if path_phone_labels is not None:
            self.use_phone_labels(path_phone_labels)
        if path_snr_labels is not None:
            self.use_snr_labels(path_snr_labels)
        if path_reverb_labels is not None:
            self.use_snr_labels(path_reverb_labels)

    @property
    def path_rttm(self) -> Path:
        return self.root / "rttm_files"

    @property
    def path_training_set(self) -> Path:
        return self.root / "training_set.txt"

    @property
    def path_test_set(self) -> Path:
        return self.root / "test_set.txt"

    @property
    def path_phone_labels(self) -> Path:
        return self.root / "phone_labels.txt"

    @property
    def path_snr_labels(self) -> Path:
        return self.root / "snr_labels.txt"

    @property
    def path_detailed_snr_labels(self) -> Path:
        return self.root / "detailed_snr_labels"

    @property
    def path_reverb_labels(self) -> Path:
        return self.root / "reverb_labels.txt"

    @property
    def path_transcription(self) -> Path:
        return self.root / "transcriptions.txt"

    @property
    def path_16k(self) -> Path:
        return self.root / "audio_16k"

    @property
    def path_pyannote_dir(self) -> Path:
        return self.root / "pyannote"

    @property
    def path_proba_pyannote(self) -> Path:
        return self.path_pyannote_dir / "proba"

    @property
    def path_complete_rttm_file(self) -> Path:
        return self.path_pyannote_dir / "rttm"

    def get_path_pyannote_file(self, subset: str, extension: str) -> Path:
        return self.path_pyannote_dir / f"{self.dataset_name}.{subset}{extension}"

    def use_training_set(self, path_training_set: Union[Path, str]) -> None:
        assert Path(path_training_set).is_file()
        shutil.copy(path_training_set, self.path_training_set)
        print(f"Training set copied from {path_training_set}")

    def use_test_set(self, path_test_set: Union[Path, str]) -> None:
        assert Path(path_test_set).is_file()
        shutil.copy(path_test_set, self.path_test_set)
        print(f"Test set copied from {path_test_set}")

    def use_phone_labels(self, path_phone_labels) -> None:
        assert Path(path_phone_labels).is_file()
        shutil.copy(path_phone_labels, self.path_phone_labels)
        print(f"Phone labels copied from {path_phone_labels}")

    def use_snr_labels(self, path_snr_labels) -> None:
        assert Path(path_snr_labels).is_file()
        shutil.copy(path_snr_labels, self.path_snr_labels)
        print(f"SNR labels copied from {path_snr_labels}")

    def get_all_files(self) -> List[Path]:
        return [x for x in self.path_16k.glob(f"**/*{self.ext_db}")]

    def load_voice_activity(self) -> Dict[str, List[Tuple[float]]]:

        if self.path_rttm.is_dir():
            file_names = self.get_all_files()
            file_data = {
                str(x): load_speech_activities_from_rttm(
                    self.path_rttm / f"{x.stem}_vad.rttm"
                )
                for x in file_names
            }
        else:
            file_data = {
                self.path_16k / x: val
                for x, val in load_int_sequences(self.path_phone_labels).items()
            }

        return file_data

    def save_voice_activity(self, data: Dict[str, List[Tuple[float]]]) -> None:
        print(f"Saving the speech activity to {self.path_rttm}")
        self.path_rttm.mkdir(exist_ok=True)
        for x, value in data.items():
            save_speech_activities_to_rttm(
                value, self.path_rttm / f"{Path(x).stem}.rttm"
            )

    def save_reverb_labels(self, data: Dict[str, List[Tuple[float, str]]]) -> None:
        print(f"Saving the reverberation information to {self.path_reverb_labels}")
        with open(self.path_reverb_labels, "w") as f_:
            for x, value in data.items():
                c50_value, path_ir = value
                f_.write(f"{Path(x).stem} {c50_value} {path_ir}\n")

    def save_snr_labels(self, data: Dict[str, float]) -> None:
        print(f"Saving the SNR information to {self.path_snr_labels}")
        with open(self.path_snr_labels, "w") as f_:
            for file_path, snr in data.items():
                f_.write(f"{Path(file_path).stem} {snr}\n")

    def save_detailed_snr_labels(self, data: Dict[str, List[List[float]]]) -> None:
        print(f"Saving the detailed SNR information to {self.path_detailed_snr_labels}")
        self.path_detailed_snr_labels.mkdir(exist_ok=True)
        for file_path, value in data.items():
            save_detailed_snr_labels(
                value, self.path_detailed_snr_labels / f"{Path(file_path).stem}_snr.npy"
            )

    def create_phone_labels(self, *args) -> None:
        raise RuntimeError("create_phone_labels is not implemented")

    def create_rttm(self, *args) -> None:
        raise RuntimeError("create_rttm is not implemented")

    def build_from_root_dir(
        self,
        path_root: Union[Path, str],
        original_extension: str = ".wav",
        n_process: int = 2,
        **kwargs,
    ) -> None:
        self.resample(path_root, original_extension, n_process)

    # resample audio files to sr_db
    def resample(
        self,
        path_original_audio: Union[Path, str],
        original_extension: str = ".wav",
        n_process: int = 2,
    ):
        self.path_16k.mkdir(exist_ok=True)
        path_original_audio = Path(path_original_audio)

        files = Path(path_original_audio).glob(f"**/*{original_extension}")

        to_deal = []
        for rel_path in files:
            pathIn = path_original_audio / rel_path
            pathOut = (self.path_16k / rel_path.name).with_suffix(self.ext_db)
            pathOut.parent.mkdir(exist_ok=True, parents=True)
            to_deal.append((str(pathIn), str(pathOut), self.sr_db))

        with Pool(n_process) as pool:
            for _ in tqdm(
                pool.imap_unordered(resample_file, to_deal, chunksize=10),
                total=len(to_deal),
            ):
                pass

        print("Resampling done")

    def build_train_test_sets(self, share_train: float = 0.9) -> None:
        files = find_seqs_relative(
            self.path_16k, extension=self.ext_db, load_cache=not self.ignore_cache
        )
        print(f"{len(files)} files found")

        # TODO Upgrade into speaker split
        shuffle(files)
        lim = int(share_train * len(files))

        save_sequence_file(files[:lim], self.path_training_set)
        print(f"Training set saved at {self.path_training_set}")

        save_sequence_file(files[lim:], self.path_test_set)
        print(f"Test set saved at {self.path_test_set}")

    # convert phone labels into rttm files
    def create_rttm_from_phone_labels(self):
        phone_labels = load_int_sequences(self.path_phone_labels)

        for rel_path, file_phones in tqdm(phone_labels.items()):
            path_rttm_out = (self.path_rttm / rel_path).with_suffix(".rttm")
            path_rttm_out.parent.mkdir(exist_ok=True, parents=True)
            build_rttm_file_from_phone_labels(file_phones, path_rttm_out)
        print(f"RTTM files saved at {self.path_rttm}")

    # convert rttm files into phone labels

    def extract_phone_labels_from_rttm(self):
        files = find_seqs_relative(
            self.path_rttm, extension=".rttm", load_cache=not self.ignore_cache
        )
        print(f"{len(files)} RTTM files found")

        with open(self.path_phone_labels, "w") as phone_labels_file:
            for rel_path in tqdm(files):
                path_rttm_file = self.path_rttm / rel_path
                filename = os.path.splitext(rel_path)[0]
                labels = speech_activities_to_int_sequence(
                    load_speech_activities_from_rttm(path_rttm_file)
                )
                labels_str = " ".join(map(str, labels))
                phone_labels_file.write(f"{filename} {labels_str}\n")
        print(f"Phone labels saved at {self.path_phone_labels}")

    # check that the training set and phone labels files contain the same audio files
    # we delete files of the training set that are not in the phone label file

    def check(self):
        with open(self.path_training_set, "r") as f:
            training_set_snr = f.readlines()
        with open(self.path_phone_labels, "r") as f:
            phone_labels_snr = f.readlines()

        training_set_snr = [x.replace("\n", "") for x in training_set_snr]
        phone_labels_snr = [x.split()[0] for x in phone_labels_snr]

        with open(self.path_training_set, "w") as training_set_snr_w:
            print("Files in training_set but not in phone_labels:")
            for file in training_set_snr:
                if not file in phone_labels_snr:
                    print(file)
                else:
                    training_set_snr_w.write(f"{file}\n")

        print("Files in phone_labels but not in training_set:")
        for file in phone_labels_snr:
            if not file in training_set_snr:
                print(file)
        return

    # to run pyannote, we need three files: .lst with the list of audio files to parse,
    # .uem with the duration of audio files, and .rttm that gathers all rttm files

    def create_dataset_pyannote(self, path_out: Union[Path, str]):
        path_out = Path(path_out)
        self.path_pyannote_dir.mkdir(exist_ok=True)
        lst_test = open(
            self.get_path_pyannote_file("test", ".lst"),
            "w",
        )
        rttm_test = open(
            self.get_path_pyannote_file("test", ".rttm"),
            "w",
        )
        uem_test = open(
            self.get_path_pyannote_file("test", ".uem"),
            "w",
        )

        with open(self.path_training_set) as f:
            files = [x.strip() for x in f.readlines()]

        for file in tqdm(files):
            with open(os.path.join(self.path_rttm, file + ".rttm")) as f:
                rttm_tmp = f.readlines()

            if len(rttm_tmp) == 0:
                continue

            end_time = float(rttm_tmp[len(rttm_tmp) - 1].split()[3]) + float(
                rttm_tmp[len(rttm_tmp) - 1].split()[4]
            )
            uem_test.write(f"{file} 1 0.000 {end_time}\n")

            for rttm_line in rttm_tmp:
                rttm_line_split = rttm_line.split()
                rttm_line_split[1] = file
                rttm_test.write(" ".join(rttm_line_split) + "\n")

            lst_test.write(f"{file}\n")

        lst_test.close()
        rttm_test.close()
        uem_test.close()
        print(
            f"Dataset files (.lst, .uem, .rttm) saved at \
                {self.path_pyannote_dir}"
        )

    # create a file with the gold labels for pyannote for a given file
    def __create_gold_pyannote_file(self, file: Union[Path, str]) -> List[float]:
        r"""This function converts rttm files into a gold label file
        with the same format as pyannote probability files."""
        path_proba = self.path_proba_pyannote / file
        if not os.path.exists(path_proba):
            return None

        with open(path_proba) as f_proba:
            nb_label = len(f_proba.readlines())

        path_rttm = os.path.join(self.path_rttm, os.path.splitext(file)[0] + ".rttm")
        with open(path_rttm) as f:
            content = f.readlines()

        if len(content) == 0:
            return []

        timeline = [0] * nb_label
        for line in content:
            line_split = line.split()
            # pyannote computes probabilities of speech every
            # TIME_STEP_PYANNOTE
            start_time = float(line_split[3]) / TIME_STEP_PYANNOTE
            end_time = (
                float(line_split[3]) + float(line_split[4])
            ) / TIME_STEP_PYANNOTE
            # we set the gold labels of speech regions to 1
            timeline[int(start_time) : int(end_time)] = [1] * (
                int(end_time) - int(start_time)
            )
        return timeline

    # create files with gold labels for pyannote
    def create_gold_pyannote(self):
        files = find_seqs_relative(
            self.path_proba_pyannote, extension=".txt", loadCache=not self.ignore_cache
        )
        print(f"{len(files)} files found")

        for file in tqdm(files):
            labels = self.__create_gold_pyannote_file(file)
            if not labels:
                continue

            os.makedirs(
                os.path.join(
                    self.path_proba_pyannote.replace("proba", "gold"),
                    os.path.dirname(file),
                ),
                exist_ok=True,
            )
            np.savetxt(
                os.path.join(self.path_proba_pyannote.replace("proba", "gold"), file),
                labels,
            )
        print(
            f"Gold labels files saved at {self.path_proba_pyannote.replace('proba', 'gold')}"
        )
