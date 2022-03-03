# import pyximport
# pyximport.install()

from copy import deepcopy
from pathlib import Path
from multiprocessing import Pool
import torchaudio
import torch
from typing import List, Union
from cpc_dataset_maker.vad_pyannote.vad_pyx.vad_squasher import get_intervals_from_proba

class VADFeeder:
    def __init__(
        self,
        seq_names: List[str],
        path_db: Union[Path, str],
        size_frame: int,
        size_output_vad: int,
        n_process_loader: int = 1,
    ):

        self._seq_names = deepcopy(seq_names)
        self._path_db = Path(path_db)
        self._size_frame = size_frame
        self._size_vad_chunk = size_output_vad
        self.n_process_loader = n_process_loader

        with Pool(self.n_process_loader) as p:
            self._n_chunk_seq = torch.tensor(
                p.map(self.get_n_chunks_sequence, list(range(len(seq_names)))),
                dtype=torch.long,
            )

        self._cum_size_chunks = torch.cat(
            (torch.zeros(1, dtype=torch.long), torch.cumsum(self._n_chunk_seq, 0)),
            dim=0,
        )
        self.setup_vad_vector()

    def setup_vad_vector(self):

        n_chunks = self._cum_size_chunks[-1]
        self._vad_vector = torch.zeros(n_chunks, self._size_vad_chunk, 2)

    def get_n_chunks_sequence(self, seq_index: int):
        info = torchaudio.info(str(self._path_db / self._seq_names[seq_index]))[0]
        n_frames = info.length
        return n_frames // self._size_frame

    def feed_seq_data(self, seq_index: int, chunk_index: int, vad_data: torch.Tensor):
        index_chunk = self._cum_size_chunks[seq_index] + chunk_index
        self._vad_vector[index_chunk] = vad_data

    def get_vad(self, index_seq: int, dim: int = 1):

        start = self._cum_size_chunks[index_seq]
        end = self._cum_size_chunks[index_seq + 1]
        vad_data = self._vad_vector[start:end, :, dim]
        vad_data = vad_data.view((end - start) * self._size_vad_chunk)
        return vad_data

    def get_seq_name(self, index: int):
        return self._seq_names[index]

    @property
    def n_chunks(self):
        return self._cum_size_chunks[-1]

    @property
    def n_seqs(self):
        return len(self._seq_names)

    @property
    def size_vad_chunk(self):
        return self._size_vad_chunk

    @property
    def size_vad(self):
        return self._vad_vector.size()


# def get_intervals_from_proba(
#     vad_vector, time_chunk, offset_time_chunk, onset, offset, pad_start, pad_end
# ):

#     status = vad_vector[0] > onset
#     start = offset_time_chunk
#     out = []

#     for index, vad in enumerate(vad_vector[1:], 1):
#         if status:
#             if vad < offset:
#                 curr_end = index * time_chunk + pad_end + offset_time_chunk
#                 if len(out) > 0 and start <= out[-1][1]:
#                     out[-1][1] = curr_end
#                 else:
#                     out.append([start, curr_end])
#                 status = False

#         # currently inactive
#         else:
#             # switching from inactive to active
#             if vad > onset:
#                 start = index * time_chunk - pad_start + offset_time_chunk
#                 status = True
#     if status:
#         curr_end = len(vad_vector) * time_chunk + offset_time_chunk
#         if len(out) > 0 and start <= out[-1][1]:
#             out[-1][1] = curr_end
#         else:
#             out.append([start, curr_end])

#     return out


def remove_short_seq(vad_segments, min_size):

    out = []
    for start, end in vad_segments:
        if end - start > min_size:
            out.append([start, end])

    return out


def merge_short_voids(vad_segments, min_size):

    out = []
    last_void_start = 0
    for start, end in vad_segments:
        if start - last_void_start < min_size:
            if len(out) == 0:
                out = [[last_void_start, end]]
            else:
                out[-1][1] = end
        else:
            out.append([start, end])
        last_void_start = end
    return out


def build_vad_intervals(
    vad_vector,
    time_chunk,
    onset,
    offset,
    offset_time_chunk=0,
    pad_start=0,
    pad_end=0,
    min_size_sil=0,
    min_size_voice=0,
):

    vad_intervals = get_intervals_from_proba(
        vad_vector, time_chunk, offset_time_chunk, onset, offset, pad_start, pad_end
    )
    if min_size_voice > 0:
        vad_intervals = remove_short_seq(vad_intervals, min_size_voice)

    if min_size_sil > 0:
        vad_intervals = merge_short_voids(vad_intervals, min_size_sil)

    return vad_intervals
