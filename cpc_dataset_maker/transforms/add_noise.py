from copy import deepcopy
import random
from copy import deepcopy
from pathlib import Path
import torch
import torchaudio
import tqdm
from cpc_dataset_maker.transforms.transform import Transform
from cpc_dataset_maker.transforms.labels import SNR_LABEL, RMS_LABEL
from cpc_dataset_maker.transforms.extend_silences import make_ramp
from cpc_dataset_maker.transforms.normalization import (
    energy_normalization,
    peak_normalization,
)
from typing import Any, Dict, Set, Tuple, Union
from time import time

SNR_NO_NOISE = 30
CROSSFADE_MAX = 50


# add noise to audio files
class AddNoise(Transform):
    def __init__(
        self,
        dir_noise: Union[str, Path],
        ext_noise: str = ".flac",
        snr_min: float = 0.1,
        snr_max: float = 0.9 * SNR_NO_NOISE,
        snr_no_noise: float = SNR_NO_NOISE,
    ):
        self.dir_noise = Path(dir_noise)
        self.noise_files = [
            str(x) for x in self.dir_noise.glob(f"**/*{ext_noise}")
        ]
        self.snr_no_noise = snr_no_noise
        self.ext_noise = ext_noise  # extension of noise sequences

        self.snr_min = snr_min  # minimum value for SNR
        self.snr_max = snr_max  # maximum value for SNR
        print(
            f"Add noise to audio files with a random SNR between {snr_min} and {snr_max}"
        )
        self.load_noise_db()

    def load_noise_db(self, crossfade_sec = 0.5, sample_rate = 16000) -> None:
        start = time()
        print("Loading the noise dataset")
        crossfade_frame = int(crossfade_sec * sample_rate)
        noise_data = []
        random.shuffle(self.noise_files)
        previous_fade_end = torch.zeros(crossfade_frame)
        for x in tqdm.tqdm(self.noise_files, total=len(self.noise_files)):
            noise_file = energy_normalization(torchaudio.load(x)[0].mean(dim=0))

            fade_begin = make_ramp(noise_file[:crossfade_frame], 0, 1)
            fade_begin = fade_begin + previous_fade_end
            noise_data.append(fade_begin)

            noise_data.append(noise_file[crossfade_frame:-crossfade_frame])

            previous_fade_end = make_ramp(noise_file[-crossfade_frame:], 1, 0)
        noise_data.append(noise_file[-crossfade_frame:])

        self.noise_data = torch.cat(noise_data, dim=0)
        end = time()
        print(f"Took {end-start} seconds to load the noise dataset.")
        print("Dataset loaded")

    @property
    def input_labels(self) -> Set[str]:
        return set()

    @property
    def output_labels(self) -> Set[str]:
        return {RMS_LABEL, SNR_LABEL}

    @property
    def size_noise(self) -> int:
        return len(self.noise_data)

    @property
    def init_params(self) -> Dict[str, Any]:
        return {
            "ext_noise": self.ext_noise,
            "dir_noise": str(self.dir_noise),
            "noise_files": self.noise_files,
            "snr_min": self.snr_min,
            "snr_max": self.snr_max,
            "snr_no_noise": self.snr_no_noise,
        }

    def __call__(
        self, audio_data: torch.tensor, sr: int, label_dict: Dict[str, Any]
    ) -> Tuple[torch.tensor, Dict[str, Any]]:

        audio_nb_frames = audio_data.size(0)

        # set a random SNR
        snr = random.random() * (self.snr_max - self.snr_min) + self.snr_min
        if snr >= self.snr_no_noise:
            snr = self.snr_no_noise
        a = float(snr) / 20
        noise_rms = 1 / (10 ** a)

        # if noise sequences are shorter than the audio file, add different
        # noise sequences one after the other
        frame_start = random.randint(0, self.size_noise - audio_nb_frames)
        noise_seq_torch = self.noise_data[frame_start : frame_start + audio_nb_frames]

        y = peak_normalization(
            energy_normalization(audio_data)
            + energy_normalization(noise_seq_torch) * noise_rms
        )
        new_labels = deepcopy(label_dict)
        new_labels[RMS_LABEL] = noise_rms
        new_labels[SNR_LABEL] = snr

        return y, new_labels
