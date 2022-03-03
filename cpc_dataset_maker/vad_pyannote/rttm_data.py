from pathlib import Path
from typing import List, Tuple, Union

CPC_SPLIT_LENGTH = 0.01


def speech_activities_to_int_sequence(
    speech_activity: List[Tuple[float]], size_audio: float
) -> List[int]:
    out = []
    last_end = 0
    for start, end in speech_activity:
        out += [0] * int((start - last_end) / CPC_SPLIT_LENGTH)
        out += [1] * int((end - start) / CPC_SPLIT_LENGTH)
        last_end = end

    out += [0] * int((size_audio - last_end) / CPC_SPLIT_LENGTH)
    return out

def build_rttm_file_from_phone_labels(
    phone_label: List[int], path_rttm_out: Union[Path, str]
):
    seq_name = Path(path_rttm_out).stem
    with open(path_rttm_out, "w") as rttm_file:
        start_time_speech = None
        for index, value in enumerate(phone_label):
            if value == 0:
                if start_time_speech is None:
                    continue
                duration = index * CPC_SPLIT_LENGTH - start_time_speech
                rttm_file.write(
                    f"SPEAKER {seq_name} 1 {start_time_speech} {duration} <NA> <NA> A <NA> <NA>\n"
                )
                start_time_speech = None
            elif start_time_speech is None:
                start_time_speech = index * CPC_SPLIT_LENGTH

        if start_time_speech is not None:
            duration = len(phone_label) * CPC_SPLIT_LENGTH - start_time_speech
            rttm_file.write(
                f"SPEAKER {seq_name} 1 {start_time_speech} {duration} <NA> <NA> A <NA> <NA>\n"
            )


def build_phone_labels_file_from_rttm_file(
    path_rttm_file: Union[str, Path], lenght_audio: float
) -> None:
    r"""Phone labels are a series of 0 (non-speech)
    and 1 (speech) every 10 ms."""

    with open(path_rttm_file, "r") as f_:
        content = f_.readlines()

    phone_labels = [0] * int(lenght_audio * 100)

    for line in content:
        line_split = line.split()
        start_time = float(line_split[3])
        lenght_audio = float(line_split[3]) + float(line_split[4])
        # set labels of speech regions to 1
        phone_labels[int(start_time * 100) : int(lenght_audio * 100)] = [1] * (
            int(lenght_audio * 100) - int(start_time * 100)
        )
    return phone_labels


def save_speech_activities_to_rttm(
    speech_activity: List[Tuple[float]], path_out: Union[Path, str]
) -> None:

    seq_name = Path(path_out).stem
    with open(path_out, "w") as rttm_file:
        for start, end in speech_activity:
            rttm_file.write(
                f"SPEAKER {seq_name} 1 {start} {end-start} <NA> <NA> A <NA> <NA>\n"
            )


def load_speech_activities_from_rttm(path_rttm : Union[Path, str] ) -> List[Tuple[float]]:

    with open(path_rttm, 'r') as f_:
        data = [x.strip() for x in f_.readlines()]

    out = []
    for line in data:
        if len(line)==0:
            continue
        vals = line.split()
        assert(vals[0] == "SPEAKER")
        assert(len(vals) == 10)

        start, duration = float(vals[3]), float(vals[4])
        out += [(start, start + duration)]

    return out