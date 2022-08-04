from copy import deepcopy
import os
import random
import numpy as np
import torch
import math
from typing import Any, Dict, List, Optional, Tuple, Union, Set
from pathlib import Path
from cpc_dataset_maker.transforms.transform import Transform
from cpc_dataset_maker.transforms.labels import SPEECH_ACTIVITY_LABEL


def make_ramp(
    audio_data: torch.tensor, multiplier_start: float, multiplier_end: float
) -> torch.tensor:

    n_steps = audio_data.size(0)
    if n_steps <= 1:
        return audio_data

    step_size = (multiplier_end - multiplier_start) / n_steps

    ramp = torch.arange(
        multiplier_start,
        multiplier_end,
        step_size,
        device=audio_data.device,
        dtype=audio_data.dtype,
    )

    return audio_data * ramp


def check_sil_seq(silences: List[Tuple[int, int]], crossfade_frame):

    if min(x[1] for x in silences) < crossfade_frame:
        raise RuntimeError("Silences must be longer than the crossfade")

    if sum(1 for x, _ in silences if x < 0) > 1:
        raise RuntimeError("You can only have one silence before the sequence")


def add_crossfading(data: torch.Tensor, crossfade_frame: int, start=True, end=True):
    if start:
        min_crossfading = len(data[0:crossfade_frame])
        data[0:crossfade_frame] = torch.linspace(0, 1, min(crossfade_frame, min_crossfading)) * data[0:crossfade_frame]
    if end:
        min_crossfading = len(data[-crossfade_frame:])
        data[-crossfade_frame:] = torch.linspace(1, 0, min(crossfade_frame, min_crossfading)) * data[-crossfade_frame:]
    return data


def add_silences_to_speech_mono(
    audio_data: torch.tensor, silences: List[Tuple[int, int]], crossfade_frame: int
) -> torch.tensor:

    check_sil_seq(silences, crossfade_frame)

    silences.sort(key=lambda x: x[0])
    out = []
    previous_start = 0
    start_crossfading = False
    end_crossfading = True

    if silences[0][0] < 0:
        out.append(torch.zeros(silences[0][1]))
        silences = silences[1:]
        start_crossfading = True

    for silence_start, duration in silences:
        audio_chunk = audio_data[previous_start:silence_start]
        audio = add_crossfading(audio_chunk, crossfade_frame, start_crossfading, end_crossfading)
        out.append(audio)
        out.append(torch.zeros(duration))
        start_crossfading = True
        previous_start = silence_start
    
    audio_chunk = audio_data[previous_start:]
    out.append(add_crossfading(audio_chunk, crossfade_frame, True, False))


    return torch.cat(out, dim=0)


def update_speech_activity_from_new_silence(
    old_speech_activity: List[Tuple[float, float]], silences: List[Tuple[float, float]]
) -> List[Tuple[float, float]]:

    new_speech_activity = old_speech_activity.copy()
    for silence_start, duration in silences:
        speech_activity_iter = iter(old_speech_activity)
        
        # let's find the speech activity index immediately after the silence insertion
        shift_index = 0
        while (speech_activity := next(speech_activity_iter, None)) and speech_activity[0] < silence_start: # affectation only in python > 3.8
            shift_index +=1

        # speech activity after the silence insertion is shifted by the duration of the silence
        new_speech_activity[shift_index:] = [(start + duration, end + duration) for (start, end) in new_speech_activity[shift_index:]]

    return new_speech_activity


def draw_sil(
    n_frames_in: int,
    proba_off: float,
    n_frames_crossfade: int,
    mean_sil: int,
    std_sil: int,
) -> List[Tuple[int, int]]:

    switches = int(n_frames_in // (2 * n_frames_crossfade))
    switched_off = (
        (torch.rand(switches) < proba_off).nonzero(as_tuple=False)
        * 2
        * n_frames_crossfade
    )
    sils = (
        torch.nn.functional.relu(
            torch.randn(switched_off.size(), dtype=float) * std_sil + mean_sil
        )
        + 2 * n_frames_crossfade
    )

    return [(int(pos.item()), int(sil.item())) for pos, sil in zip(switched_off, sils)]


def merge_sils(
    silences_duration: List[Tuple[Union[int, float], Union[int, float]]], crossfade: float
) -> List[Tuple[Union[int, float], Union[int, float]]]:

    if len(silences_duration) == 0:
        return []

    silences_duration.sort()

    out = [silences_duration[0]]
    last_start = out[0][0]
    last_end = last_start + out[0][1]
    for start, duration in silences_duration[1:]:

        if start <= last_end + 2 * crossfade:
            last_end = start + duration
            out[-1] = (last_start, last_end - last_start)
        else:
            out.append((start, duration))
            last_start, last_end = start, start + duration

    return out


def draw_sil_from_non_speech_regions(
    old_speech_activity: List[Tuple[float, float]],
    cossfade_sec: float,
    sil_mean_sec: float,
    std_sil_sec: float,
    target_share_sil: float,
    silence_min_sec: float
) -> List[Tuple[float, float]]:

    possible_insertions = [-1]
    size_audio = sum(end - start for start, end in old_speech_activity)
    last_end = 0
    for start, end in old_speech_activity:
        if start - last_end > silence_min_sec:
            possible_insertions.append(start)
        last_end = end

    possible_insertions.append(old_speech_activity[-1][-1])

    n_samples = int(size_audio * target_share_sil / sil_mean_sec)
    if n_samples >= len(possible_insertions):
        sampled_items = possible_insertions
        n_samples = len(possible_insertions)
    else:
        sampled_items = random.sample(possible_insertions, n_samples)

    silences_duration = (
        torch.nn.functional.relu(
            torch.randn(n_samples, dtype=float) * std_sil_sec + sil_mean_sec
        )
        + 2 * cossfade_sec
    )
    return [(start, duration.item()) for start, duration in zip(sampled_items, silences_duration)]


def expand_audio_and_timeline(
    audio_data: torch.tensor,
    old_speech_activity: List[Tuple[float, float]],
    sample_rate: int,
    cossfade_sec: float,
    sil_mean_sec: float,
    target_share_sil: float,
    sil_std_sec: float,
    expand_silence_only: bool = True,
    sil_min_sec: float = 0.5
) -> Tuple[torch.tensor, List[Tuple[float, float]]]:

    sil_mean_frame = int(sample_rate * sil_mean_sec)
    n_frames_crossfade = int(sample_rate * cossfade_sec)

    proba_off = 2 * n_frames_crossfade * target_share_sil / sil_mean_frame
    sil_std_frame = int(sample_rate * sil_std_sec)

    if expand_silence_only:
        sil_tuples_sec = merge_sils(
            draw_sil_from_non_speech_regions(
                old_speech_activity=old_speech_activity,
                cossfade_sec=cossfade_sec,
                sil_mean_sec=sil_mean_sec,
                std_sil_sec=sil_std_sec,
                target_share_sil=target_share_sil,
                silence_min_sec=sil_min_sec
            ),
            cossfade_sec,
        )
        sil_tuples_frames = [
            (int(x * sample_rate), int(dur * sample_rate)) for x, dur in sil_tuples_sec
        ]
    else:
        raise NotImplementedError(
            "Draw silences from any region is not implemented yet"
        )

    if len(sil_tuples_frames) == 0:
        return audio_data, old_speech_activity

    new_audio = add_silences_to_speech_mono(
        audio_data, sil_tuples_frames, n_frames_crossfade
    )
    new_speech_activity = update_speech_activity_from_new_silence(
        old_speech_activity, sil_tuples_sec
    )

    return new_audio, new_speech_activity


class ExtendSilenceTransform(Transform):
    def __init__(
        self,
        cossfade_sec: float,
        sil_mean_sec: float,
        target_share_sil: float,
        sil_std_sec: Optional[float] = None,
        expand_silence_only: bool = True,
        sil_min_sec: float = 0.5
    ) -> None:
        super().__init__()

        if sil_std_sec is None:
            sil_std_sec = math.sqrt(sil_mean_sec)
        self.cossfade_sec = cossfade_sec
        self.sil_mean_sec = sil_mean_sec
        self.target_share_sil = target_share_sil
        self.sil_std_sec = sil_std_sec
        self.expand_silence_only = expand_silence_only
        self.sil_min_sec = sil_min_sec

    @property
    def input_labels(self) -> Set[str]:
        return {SPEECH_ACTIVITY_LABEL}

    @property
    def output_labels(self) -> Set[str]:
        return {SPEECH_ACTIVITY_LABEL}

    @property
    def init_params(self) -> Dict[str, Any]:
        return {
            "cossfade_sec": self.cossfade_sec,
            "sil_mean_sec": self.sil_mean_sec,
            "target_share_sil": self.target_share_sil,
            "sil_std_sec": self.sil_std_sec,
            "expand_silence_only": self.expand_silence_only,
        }

    def __call__(
        self, audio_data: torch.tensor, sr: int, label_dict: Dict[str, Any]
    ) -> Tuple[torch.tensor, Dict[str, Any]]:

        new_audio, new_speech_activity = expand_audio_and_timeline(
            audio_data=audio_data,
            old_speech_activity=label_dict[SPEECH_ACTIVITY_LABEL],
            sample_rate=sr,
            cossfade_sec=self.cossfade_sec,
            sil_mean_sec=self.sil_mean_sec,
            target_share_sil=self.target_share_sil,
            sil_std_sec=self.sil_std_sec,
            expand_silence_only=self.expand_silence_only,
            sil_min_sec=self.sil_min_sec
        )

        new_labels = deepcopy(label_dict)
        new_labels[SPEECH_ACTIVITY_LABEL] = new_speech_activity

        return new_audio, new_labels
