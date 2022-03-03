import torchaudio
import torch
from pathlib import Path
from copy import deepcopy
from multiprocessing import Pool
from typing import List, Tuple, Union


class SeqNameDataset:
    def __init__(
        self,
        path_db: Union[str, Path],
        seq_names: List[str],
        size_frame: int,
        size_batch: int,
        n_process_loader: int = 10,
        MAX_SIZE_LOADED: int = 2000000000,
    ):

        self.MAX_SIZE_LOADED = MAX_SIZE_LOADED
        self.n_process_loader = n_process_loader
        self._path_db = Path(path_db)
        self._size_frame = size_frame
        self._seq_names = deepcopy(seq_names)
        self.batch_size = size_batch

        self.prepare()

    def get_n_batches_from_size(self, i_size: int) -> int:
        n_group = i_size // self._size_frame
        n_batch = n_group // self.batch_size
        if n_batch % self.batch_size > 0:
            n_batch += 1
        return n_batch

    def load_seq(
        self, seq_index: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        data = torchaudio.load(str(self._path_db / self._seq_names[seq_index]))[0].mean(
            dim=0
        )
        S = data.size(0)
        cut = S - (S % self._size_frame)
        data = data[:cut].view(-1, self._size_frame, 1)
        n_cut = data.size(0)
        # if n_cut > 0:
        #     data = data / data.abs().max()
        return (
            data,
            seq_index * torch.ones(n_cut, dtype=torch.long),
            torch.arange(n_cut, dtype=torch.long),
        )

    def extract_length(self, seq_index: int) -> int:
        info = torchaudio.info(str(self._path_db / self._seq_names[seq_index]))[0]
        return info.length - (info.length % self._size_frame)

    def prepare(self) -> None:

        self._load_indexes = []
        self._len = 0
        with torch.no_grad():
            with Pool(self.n_process_loader) as p:
                all_length = p.map(self.extract_length, list(range(self.n_seqs)))

        start, curr_size = 0, all_length[0]
        for index, size in enumerate(all_length[1:], 1):
            if curr_size > self.MAX_SIZE_LOADED:
                self._load_indexes.append((start, index))
                self._len += self.get_n_batches_from_size(curr_size)
                curr_size = 0
                start = index

            curr_size += size

        if curr_size > 0:
            self._load_indexes.append((start, len(all_length)))
            self._len += self.get_n_batches_from_size(curr_size)

    def load_segment(self, index_segment: int):

        self._frame_data = []
        self._index_data = []
        self._chunk_index = []

        start, end = self._load_indexes[index_segment]
        with torch.no_grad():
            with Pool(self.n_process_loader) as p:
                curr_pack = p.map(self.load_seq, list(range(start, end)))

        for data, seq_index, chunk_index in curr_pack:

            self._frame_data.append(data)
            self._index_data.append(seq_index)
            self._chunk_index.append(chunk_index)

        self._frame_data = torch.cat(self._frame_data, dim=0)
        self._index_data = torch.cat(self._index_data, dim=0)
        self._chunk_index = torch.cat(self._chunk_index, dim=0)

        assert self._frame_data.size(0) == self._index_data.size(0)
        assert self._frame_data.size(0) == self._chunk_index.size(0)

    def __iter__(self):

        n_reloads = len(self._load_indexes)
        for reload_index in range(n_reloads):
            self.load_segment(reload_index)

            n_batches = self._frame_data.size(0)
            # self._frame_data = self._frame_data.cuda(non_blocking=True)
            size_loop = n_batches // self.batch_size
            if n_batches % self.batch_size > 0:
                size_loop += 1

            for i_batch in range(size_loop):
                index_start = self.batch_size * i_batch
                index_end = min(n_batches, index_start + self.batch_size)
                size = index_end - index_start

                yield self._frame_data[index_start:index_end].view(
                    size, -1, 1
                ), self._index_data[index_start:index_end], self._chunk_index[
                    index_start:index_end
                ]

    def __len__(self) -> int:
        return self._len

    @property
    def n_reloads(self) -> int:
        return len(self._load_indexes)

    @property
    def n_seqs(self) -> int:
        return len(self._seq_names)
