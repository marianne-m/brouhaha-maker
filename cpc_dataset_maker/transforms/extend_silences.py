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
    ramp = multiplier_end * torch.ones(
        audio_data.size(), device=audio_data.device, dtype=audio_data.dtype
    )
    ramp[:n_steps] = torch.arange(
        multiplier_start,
        multiplier_end,
        step_size,
        device=audio_data.device,
        dtype=audio_data.dtype,
    )[:n_steps]

    return audio_data * ramp


def check_sil_seq(silences: List[Tuple[int, int]], crossfade_frame):

    if min(x[1] for x in silences) < crossfade_frame:
        raise RuntimeError("Silences must be longer than the crossfade")

    if sum(1 for x, _ in silences if x < 0) > 1:
        raise RuntimeError("You can only have one silence before the sequence")


def add_silences_to_speech_mono(
    audio_data: torch.tensor, silences: List[Tuple[int, int]], crossfade_frame: int
) -> torch.tensor:

    check_sil_seq(silences, crossfade_frame)

    silences.sort(key=lambda x: x[0])
    out = []
    last_frame_end = 0
    ramp_last = False

    if silences[0][0] < 0:
        duration = silences[0][1]
        out += [torch.zeros(duration - crossfade_frame, dtype=torch.float)]
        silences = silences[1:]
        ramp_last = True

    for frame_start, duration in silences:
        shift = 0
        if ramp_last:
            out += [
                make_ramp(
                    audio_data[last_frame_end : last_frame_end + crossfade_frame], 0, 1
                )
            ]
            shift = last_frame_end + crossfade_frame
        if shift > frame_start:
            raise RuntimeError("Speech activity smaller than crossfade")
        out += [audio_data[shift:frame_start]]
        out += [
            make_ramp(audio_data[frame_start : frame_start + crossfade_frame], 1, 0)
        ]
        n_zeros = duration - out[-1].size(0)
        if frame_start + duration < audio_data.size(0):
            n_zeros -= crossfade_frame
            last_frame_end = frame_start + crossfade_frame
        else:
            last_frame_end = audio_data.size(0)
        if n_zeros < 0:
            raise RuntimeError("Crossfade larger than silence")
        out += [
            torch.zeros(
                n_zeros,
                device=audio_data.device,
                dtype=audio_data.dtype,
            )
        ]
        ramp_last = True

    if last_frame_end < audio_data.size(0):
        out += [
            make_ramp(
                audio_data[last_frame_end : last_frame_end + crossfade_frame], 0, 1
            )
        ]
        out += [audio_data[last_frame_end + crossfade_frame :]]

    return torch.cat(out, dim=0)


def update_speech_activity_from_new_silence(
    old_speech_activity: List[Tuple[float, float]], silences: List[Tuple[float, float]]
) -> List[Tuple[float, float]]:

    new_timeline = []
    offset_start, offset_end = 0, 0
    i_old_timeline = 0
    o_speech_start, o_speech_end = old_speech_activity[0]
    for sil_start, sil_duration in silences:
        while o_speech_end < sil_start:
            new_timeline += [(o_speech_start + offset_start, o_speech_end + offset_end)]
            offset_start = offset_end
            i_old_timeline += 1

            if i_old_timeline >= len(old_speech_activity):
                return new_timeline
            else:
                o_speech_start, o_speech_end = old_speech_activity[i_old_timeline]
        if sil_start <= o_speech_start:
            offset_end += sil_duration
            offset_start += sil_duration
        elif sil_start < o_speech_end:
            offset_end += sil_duration
            if o_speech_start < sil_start:
                new_timeline += [
                    (o_speech_start + offset_start, sil_start + offset_start)
                ]
            o_speech_start = sil_start
            offset_start += sil_duration
        else:
            offset_end += sil_duration
            offset_start = offset_end

    new_timeline += [(o_speech_start + offset_start, o_speech_end + offset_end)]
    offset_end = offset_start
    for start, end in old_speech_activity[i_old_timeline + 1 :]:
        new_timeline += [(start + offset_start, end + offset_end)]

    return new_timeline


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
    sils: List[Tuple[Union[int, float], Union[int, float]]], crossfade: float
) -> List[Tuple[Union[int, float], Union[int, float]]]:

    if len(sils) == 0:
        return []

    sils.sort()

    out = [sils[0]]
    last_start = out[0][0]
    last_end = last_start + out[0][1]
    for start, duration in sils[1:]:

        if start <= last_end + 2 * crossfade:
            last_end = start + duration
            out[-1] = (last_start, last_end - last_start)
        else:
            out += [(start, duration)]
            last_start, last_end = start, start + duration

    return out


def draw_sil_from_non_speech_regions(
    old_speech_activity: List[Tuple[float, float]],
    cossfade_sec: float,
    sil_mean_sec: float,
    std_sil_sec: float,
    target_share_sil: float,
) -> List[Tuple[float, float]]:

    possible_insertions = [-1]
    size_audio = sum(end - start for start, end in old_speech_activity)
    last_end = 0
    for start, end in old_speech_activity:
        if start - last_end > 2 * cossfade_sec:
            possible_insertions += [start]
        last_end = end

    possible_insertions += [old_speech_activity[-1][-1]]

    n_samples = int(size_audio * target_share_sil / sil_mean_sec)
    if n_samples >= len(possible_insertions):
        sampled_items = possible_insertions
        n_samples = len(possible_insertions)
    else:
        sampled_items = random.sample(possible_insertions, n_samples)

    sils = (
        torch.nn.functional.relu(
            torch.randn(n_samples, dtype=float) * std_sil_sec + sil_mean_sec
        )
        + 2 * cossfade_sec
    )
    return [(x, dur.item()) for x, dur in zip(sampled_items, sils)]


def expand_audio_and_timeline(
    audio_data: torch.tensor,
    old_speech_activity: List[Tuple[float, float]],
    sample_rate: int,
    cossfade_sec: float,
    sil_mean_sec: float,
    target_share_sil: float,
    sil_std_sec: float,
    expand_silence_only: bool = True,
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
            ),
            cossfade_sec,
        )
        sil_tuples_frames = [
            (int(x * sample_rate), int(dur * sample_rate)) for x, dur in sil_tuples_sec
        ]
    else:
        sil_tuples_frames = merge_sils(
            draw_sil(
                n_frames_in=audio_data.size(0),
                proba_off=proba_off,
                n_frames_crossfade=n_frames_crossfade,
                mean_sil=sil_mean_frame,
                std_sil=sil_std_frame,
            ),
            n_frames_crossfade,
        )
        sil_tuples_sec = [
            (x / sample_rate, dur / sample_rate) for x, dur in sil_tuples_frames
        ]

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
    ) -> None:
        super().__init__()

        if sil_std_sec is None:
            sil_std_sec = math.sqrt(sil_mean_sec)
        self.cossfade_sec = cossfade_sec
        self.sil_mean_sec = sil_mean_sec
        self.target_share_sil = target_share_sil
        self.sil_std_sec = sil_std_sec
        self.expand_silence_only = expand_silence_only

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
        )

        new_labels = deepcopy(label_dict)
        new_labels[SPEECH_ACTIVITY_LABEL] = new_speech_activity

        return new_audio, new_labels
