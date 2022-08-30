from copy import deepcopy
import random
from pathlib import Path
from this import d
from typing import Any, Dict, Set, List, Tuple, Union
import numpy as np
import torch
import torchaudio
from cpc_dataset_maker.transforms.transform import Transform
from cpc_dataset_maker.transforms.add_reverb import Reverb
from cpc_dataset_maker.transforms.labels import SNR_LABEL, DETAILED_SNR, RMS_LABEL, SPEECH_ACTIVITY_LABEL
from cpc_dataset_maker.transforms.extend_silences import make_ramp
from cpc_dataset_maker.transforms.normalization import (
    energy_normalization,
    energy_normalization_on_vad,
    peak_normalization,
)


SNR_NO_NOISE = 30
CROSSFADE_MAX = 0.05
SAMPLE_RATE = 16000
AUDIOSET_SIZE = 8
SAMPLE_MAX = 500
WINDOW_STEP = 0.01
WINDOW_SIZE = 2


def compute_detailed_snr(
    audio_data: torch.Tensor,
    noise: torch.Tensor,
    sample_rate: int,
    vad: List[Tuple[float, float]],
    step: float = WINDOW_STEP,
    window_size: int = WINDOW_SIZE
) -> List[List[float]]:
    """
    Compute the mean snr every step on a sliding window of size window_size.
    step and window_size are in seconds
    """
    window_frames = int(window_size * sample_rate)
    step_in_frames = int(step * sample_rate)

    vad_mask = np.zeros_like(audio_data)
    vad_frame = [(int(st*sample_rate), int(end*sample_rate)) for st, end in vad]
    for start_speech_activity, end_speech_activity in vad_frame:
        vad_mask[start_speech_activity:end_speech_activity] = 1
    audio_data = audio_data * vad_mask  

    power_audio = np.cumsum(np.square(audio_data))
    power_noise = np.cumsum(np.square(noise))

    start_windows = np.arange(0, audio_data.size(0) - window_frames + 1, step_in_frames)
    end_windows = np.arange(window_frames - 1, audio_data.size(0), step_in_frames)  

    epsilon = 10**(-3)
    # max(signal_to_noise, 10**(-10)) so the snr_db is equal to -100 when the signal is 0
    signal_to_noise = [
        max(
            (power_audio[end] - power_audio[start]) / (power_noise[end] - power_noise[start] + epsilon),
            10**(-10)
        ) for start, end in zip(start_windows, end_windows)
    ]

    snr_db = 10 * np.log10(signal_to_noise)

    return snr_db


def save_detailed_snr_labels(values: List[float], file_path: Union[Path, str]) -> None:
    with open(file_path, 'wb') as snr_file:
        np.save(snr_file, values)


# add noise to audio files
class AddNoise(Transform):
    def __init__(
        self,
        dir_noise: Union[str, Path],
        dir_impulse_response: Union[str, Path] = None,
        ext_noise: str = ".flac",
        snr_min: float = 0,
        snr_max: float = 0.9 * SNR_NO_NOISE,
        snr_no_noise: float = SNR_NO_NOISE,
        crossfading_duration: float = CROSSFADE_MAX,
        window_step: float = WINDOW_STEP,
        window_size: float = WINDOW_SIZE
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
        self.crossfading_duration = crossfading_duration
        self.window_size = window_size
        self.window_step = window_step

        # if no dir_impulse_response is given, the noise is added without reverberation
        # else, an impulse response is applied to the noise
        if dir_impulse_response :
            print("Noise dataset is applied with reverberation")
            self.reverb_transform = Reverb(dir_impulse_response)
            self.load_noise = self.load_with_reverb
        else:
            print("Noise dataset is applied without reverberation")
            self.load_noise = self.simple_load

    def load_noise_on_the_fly(
        self,
        audio_nb_frames: int,
        sample_rate: int
    ) -> torch.Tensor:
        crossfade_frame = int(self.crossfading_duration * sample_rate)
        noise_data = []
        noise_files = random.sample(self.noise_files, SAMPLE_MAX)

        previous_fade_end = torch.zeros(crossfade_frame)

        index_noise_file = 0
        noise = torch.Tensor([])
        while len(noise) < 2*audio_nb_frames:
            noise_file = self.load_noise(noise_files[index_noise_file], sample_rate)

            fade_begin = make_ramp(noise_file[:crossfade_frame], 0, 1)
            fade_begin = fade_begin + previous_fade_end
            noise_data.append(fade_begin)
            noise_data.append(noise_file[crossfade_frame:-crossfade_frame])
            previous_fade_end = make_ramp(noise_file[-crossfade_frame:], 1, 0)

            index_noise_file += 1
            noise = torch.cat(noise_data, dim=0)

        noise_data.append(noise_file[-crossfade_frame:])

        noise_data = torch.cat(noise_data, dim=0)
        return noise_data

    @property
    def input_labels(self) -> Set[str]:
        return set()

    @property
    def output_labels(self) -> Set[str]:
        return {RMS_LABEL, SNR_LABEL, DETAILED_SNR}

    @property
    def init_params(self) -> Dict[str, Any]:
        return {
            "ext_noise": self.ext_noise,
            "dir_noise": str(self.dir_noise),
            "noise_files": self.noise_files,
            "snr_min": self.snr_min,
            "snr_max": self.snr_max,
            "snr_no_noise": self.snr_no_noise,
            "crossfading_duration": self.crossfading_duration,
            "window_step": self.window_step,
            "window_size": self.window_size
        }

    def __call__(
        self, audio_data: torch.Tensor, sr: int, label_dict: Dict[str, Any]
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:

        audio_nb_frames = audio_data.size(0)

        # set a random SNR
        snr = random.random() * (self.snr_max - self.snr_min) + self.snr_min
        if snr >= self.snr_no_noise:
            snr = self.snr_no_noise
        a = float(snr) / 20
        noise_rms = 1 / (10 ** a)

        # if noise sequences are shorter than the audio file, add different
        # noise sequences one after the other
        noise = self.load_noise_on_the_fly(audio_nb_frames, sr)
        frame_start = random.randint(0, noise.size(0) - audio_nb_frames)
        noise_seq_torch = noise[frame_start : frame_start + audio_nb_frames]

        audio_data_normalized = energy_normalization_on_vad(
            audio_data,
            label_dict[SPEECH_ACTIVITY_LABEL],
            sr
        )
        noise = energy_normalization(noise_seq_torch) * noise_rms

        y = peak_normalization(
            audio_data_normalized
            + noise
        )
        new_labels = deepcopy(label_dict)
        new_labels[RMS_LABEL] = noise_rms
        new_labels[SNR_LABEL] = snr
        new_labels[DETAILED_SNR] = compute_detailed_snr(
            audio_data_normalized,
            noise,
            sr,
            label_dict[SPEECH_ACTIVITY_LABEL],
            self.window_step,
            self.window_size
        )

        return y, new_labels

    def load_with_reverb(
        self,
        waveform: torch.Tensor,
        sample_rate: int
    ) -> torch.Tensor:
        noise_file = torchaudio.load(waveform)[0].mean(dim=0)
        noise_file_reverb, _ = self.reverb_transform.__call__(
            noise_file, sample_rate, dict()
        )
        return noise_file_reverb

    @staticmethod
    def simple_load(
        waveform: torch.Tensor,
        sample_rate: int
    ) -> torch.Tensor:
        return torchaudio.load(waveform)[0].mean(dim=0)
