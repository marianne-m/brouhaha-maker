from copy import deepcopy
import enum
import json
from pathlib import Path
import torch
import torchaudio
from typing import Any, Dict, Set, List, Tuple, Union
from cpc_dataset_maker.transforms.labels import (
    SPEECH_ACTIVITY_LABEL,
    INPUT_PATH_KEY,
    OUTPUT_PATH_KEY,
    LABELS_KEY,
    FILE_ORDER_LABEL,
)
import tqdm
from torch.multiprocessing import Pool


def get_next_index(list_: List[Any], condition: Any, shift: int = 0) -> int:

    try:
        return next(i for i, x in enumerate(list_[shift:]) if condition(x)) + shift

    except StopIteration:
        return len(list_)


def update_timeline_from_segmentation(
    timeline: List[Tuple[float, float]], segmentation: List[float]
) -> List[List[Tuple[float, float]]]:

    if len(segmentation) == 0 or len(timeline) == 0:
        return [timeline]

    assert segmentation[0] > 0
    i_segment = 0
    curr_timeline = []
    out = []

    shift = 0
    i_segment = get_next_index(segmentation, lambda x: x > timeline[0][0])
    if i_segment > 0:
        shift = segmentation[i_segment - 1]
    if i_segment == len(segmentation):
        next_cut = timeline[-1][-1] + 1
    else:
        next_cut = segmentation[i_segment]

    out += [[] for _ in range(i_segment)]

    for start, end in timeline:

        if end < next_cut:
            curr_timeline += [(start - shift, end - shift)]
            continue

        if start < next_cut:
            curr_timeline += [(start - shift, next_cut - shift)]
            out += [curr_timeline]
            i_segment += 1

            while i_segment < len(segmentation) and segmentation[i_segment] < end:
                out += [[(0, segmentation[i_segment] - segmentation[i_segment - 1])]]
                i_segment += 1

            curr_timeline = [(0, end - segmentation[i_segment - 1])]
            shift = segmentation[i_segment - 1]
        else:
            out += [curr_timeline]
            next_segment = get_next_index(segmentation, lambda x: x >= start)
            out += [[] for _ in range(next_segment, i_segment)]
            i_segment = next_segment
            shift = segmentation[i_segment - 1]
            while i_segment < len(segmentation) and segmentation[i_segment] < end:
                out += [
                    [
                        (
                            start - shift,
                            segmentation[i_segment] - segmentation[i_segment - 1],
                        )
                    ]
                ]
                shift = segmentation[i_segment]
                start = shift
                i_segment += 1
            curr_timeline = [(start - shift, end - shift)]

        if i_segment == len(segmentation):
            next_cut = timeline[-1][-1] + 1
        else:
            next_cut = segmentation[i_segment]

    if len(curr_timeline) > 0:
        out += [curr_timeline]

    out += [[] for _ in range(i_segment, len(segmentation))]

    return out


def cut_audio(
    data: torch.tensor, sr: int, segmentation: List[float]
) -> List[torch.tensor]:

    out = []
    last_frame = 0
    for cut in segmentation:
        next_frame = int(cut * sr)
        out += [data[last_frame:next_frame]]
        last_frame = next_frame

    out += [data[last_frame:]]
    return out


def cut_fixed_size(
    data: torch.tensor,
    sr: int,
    speech_activity: List[Tuple[float, float]],
    target_size: float,
) -> Tuple[List[torch.tensor], List[Tuple[float, float]]]:

    total_size_sec = data.size(0) / sr
    n_steps = int(total_size_sec / target_size)

    segmentation = [(k + 1) * target_size for k in range(n_steps)]

    new_audio = cut_audio(data, sr, segmentation)
    new_speech_activity = update_timeline_from_segmentation(
        speech_activity, segmentation
    )

    return new_audio, new_speech_activity


class Segmentation:
    def __init__(self, size_segment: float) -> None:
        self.size_segment = size_segment

    @property
    def input_labels(self) -> Set[str]:
        return {SPEECH_ACTIVITY_LABEL}

    @property
    def output_labels(self) -> Set[str]:
        return {SPEECH_ACTIVITY_LABEL, FILE_ORDER_LABEL}

    @property
    def json_params(self) -> Dict[str, Any]:
        return {"name": self.__class__.__name__, "init-params": self.init_params}

    @property
    def init_params(self) -> Dict[str, Any]:
        return {"size_segment": self.size_segment}

    def save(self, path_out: Union[str, Path]) -> None:

        with open(path_out, "w") as f_:
            json.dump(self.json_params, f_, indent=2)

    def __call__(
        self, audio_data: torch.tensor, sr: int, label_dict: Dict[str, Any]
    ) -> List[Tuple[torch.tensor, Dict[str, Any]]]:

        new_audio_list, new_timelines = cut_fixed_size(
            audio_data, sr, label_dict[SPEECH_ACTIVITY_LABEL], self.size_segment
        )

        out = []
        n_output = len(new_audio_list)
        for i in range(n_output):
            new_labels = deepcopy(label_dict)
            new_labels[FILE_ORDER_LABEL] = i
            new_labels[SPEECH_ACTIVITY_LABEL] = new_timelines[i]
            out.append((new_audio_list[i], new_labels))

        return out

    def _run_on_file(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:

        path_in = data[INPUT_PATH_KEY]
        path_out = Path(data[OUTPUT_PATH_KEY])
        labels = data[LABELS_KEY]

        path_out.parent.mkdir(exist_ok=True)

        audio, sr = torchaudio.load(str(path_in))
        audio = audio.mean(dim=0)
        out_data = self.__call__(audio, sr, labels)

        out = []
        for audio, label in out_data:
            loc_path_out = (
                path_out.parent
                / f"{path_out.stem}_{label[FILE_ORDER_LABEL]}{path_out.suffix}"
            )

            torchaudio.save(str(loc_path_out), audio.view(1, -1), sr)
            out += [{OUTPUT_PATH_KEY: str(loc_path_out), LABELS_KEY: label}]

        return out

    def run_on_dataset(
        self,
        list_path_labels: List[Dict[str, Any]],
        n_process: int = 10,
        chunksize: int = 10,
    ) -> List[Dict[str, Any]]:

        out = []
        with Pool(n_process) as p:
            for x in tqdm.tqdm(
                p.imap(self._run_on_file, list_path_labels, chunksize=chunksize),
                total=len(list_path_labels),
            ):
                out += x

        return out
