import argparse
import torchaudio
import json
import sys
from progressbar import ProgressBar
from typing import NamedTuple
from pathlib import Path
from cpc.dataset import findAllSeqs


class VADSegment(NamedTuple):
    name: str
    start: float
    end: float


def save_json(data, path_out):
    with open(path_out, "w") as file:
        json.dump(data, file, indent=2)


def get_path_vad(dir_out, loc_path_seq):
    return str(
        Path(dir_out)
        / Path(loc_path_seq).parent
        / f"{Path(loc_path_seq).stem}_vad.json"
    )


def get_size_seq(path_seq):
    info = torchaudio.info(path_seq)[0]
    return info.length / info.rate


def get_vad_from_sil(size_seq, sil_seq, min_size_sil=0.3, min_size_voice=0.3):

    start_voice = 0
    out = []

    for start_sil, end_sil in sil_seq:

        if end_sil - start_sil > min_size_sil:
            if start_sil - start_voice > min_size_voice:
                out.append([start_voice, start_sil])
            start_voice = end_sil

    out.append(end_sil)

    if size_seq - start_voice > min_size_voice:
        out.append([start_voice, size_seq])

    return out


def load_vad_data(path_vad_file):

    with open(path_vad_file, "r") as file:
        lines = [x.strip() for x in file.readlines()]
    out = []
    for x in lines:
        fn, s, e = x.split()
        out.append(VADSegment(name=fn, start=float(s), end=float(e)))
    return out


def gather_sil_data(vad_data):

    vad_data.sort(key=lambda x: x.name)
    out = []
    last_name = vad_data[0].name
    curr_sil_seq = [[vad_data[0].start, vad_data[0].end]]
    for segment in vad_data:
        if segment.name != last_name:
            out.append({"name": last_name, "sil": curr_sil_seq})
            last_name = segment.name
            curr_sil_seq = [[segment.start, segment.end]]
        else:
            curr_sil_seq.append([segment.start, segment.end])

    if len(curr_sil_seq) > 0:
        out.append({"name": last_name, "sil": curr_sil_seq})
    return out


def save_vad_data(sil_data, seq_list, dir_out, dir_in):

    seq_list.sort(key=lambda x: Path(x).stem)
    sil_data.sort(key=lambda x: x["name"])

    dir_out = Path(dir_out)
    dir_in = Path(dir_in)

    index_seq = 0
    for sil_seq_data in sil_data:
        while (
            index_seq < len(seq_list)
            and Path(seq_list[index_seq]).stem < sil_seq_data["name"]
        ):
            index_seq += 1

        if (
            index_seq >= len(seq_list)
            or Path(seq_list[index_seq]).stem > sil_seq_data["name"]
        ):
            continue

        full_path_in = str(dir_in / seq_list[index_seq])
        full_path_out = get_path_vad(dir_out, seq_list[index_seq])
        Path(full_path_out).parent.mkdir(exist_ok=True, parents=True)
        size_seq = get_size_seq(full_path_in)
        save_json(get_vad_from_sil(size_seq, sil_seq_data["sil"]), full_path_out)


def parse_args(argv):

    parser = argparse.ArgumentParser(description="Zerospeech vad converter")

    parser.add_argument("path_db", type=str)
    parser.add_argument("path_sil", type=str, default=".wav")
    parser.add_argument("-o", "--output", type=str, default="coin")
    parser.add_argument("--file_extension", type=str, default=".wav")

    return parser.parse_args(argv)


if __name__ == "__main__":

    args = parse_args(sys.argv[1:])

    # Step one: load the sil data
    sil_data = load_vad_data(args.path_sil)

    # Step two: gather_them
    grouped_sil = gather_sil_data(sil_data)

    # Step three: save the vad data
    seq_list = [
        x[1] for x in findAllSeqs(args.path_db, extension=args.file_extension)[0]
    ]
    save_vad_data(grouped_sil, seq_list, args.output, args.path_db)
