import unittest
import torch
import torchaudio
import os
import shutil
from pathlib import Path
from nose.tools import eq_, ok_
from cpc_dataset_maker.vad_pyannote.vad_feeder import VADFeeder
from cpc_dataset_maker.vad_pyannote.seq_name_dataset import SeqNameDataset
from cpc_dataset_maker.vad_pyannote.vad_pyx.dtw import (
    get_intervals_from_proba,
    merge_short_voids,
    remove_short_seq,
    get_intervals_from_proba,
)

# import pyximport; pyximport.install()

TMP_DIR = ".tmp"
########################################################
# TOOLS
########################################################


def save_tmp_seq_list(seq_list, sample_rate=16000, tmp_dir=".tmp", extension=".wav"):
    Path(tmp_dir).mkdir(exist_ok=True)
    for seq_name, seq_value in seq_list:
        torchaudio.save(
            str(Path(tmp_dir) / f"{seq_name}{extension}"),
            torch.tensor(seq_value),
            sample_rate,
        )


def clean_tmp_dir(tmp_dir=".tmp"):
    shutil.rmtree(tmp_dir)


########################################################


class TestVADFeeder(unittest.TestCase):
    def setUp(self):
        # Vad chunks
        seqs = [
            ("0", [0, 0.5, 0.2, -0.5, -0.7, 0.8, 0.9, 0.0]),  # 2 x 1
            ("1", [0, 0.5, 0.2, -0.5, 0.7, 0.8, 0.9, 0.0, 0.0]),  # 3 x 1
            ("2", [0, 0.0, 0.0, -0.5, -0.4, 0.8, 0.9, 0.0]),  # 2 x 1
            ("3", [0, -0.5, 0.9, 0.0, 0.7, 0.8, 0.9, 0.0]),
        ]  # 2 x 1

        self.seq_names = [f"{x}.wav" for x, _ in seqs]
        self.path_db = TMP_DIR
        self.size_frame = 3
        self.size_vad_output = 1
        save_tmp_seq_list(seqs, tmp_dir=TMP_DIR)

        self.vad_data = VADFeeder(
            self.seq_names, self.path_db, self.size_frame, self.size_vad_output
        )

    def build_feeding_vector(self, to_feed):

        size_batch = len(to_feed)
        index_ = torch.zeros(size_batch, dtype=torch.long)
        chunk_ = torch.zeros(size_batch, dtype=torch.long)
        vad_ = torch.zeros((size_batch, 1, 2), dtype=torch.float)

        for i in range(size_batch):
            index, chunk, vad = to_feed[i]
            index_[i] = index
            chunk_[i] = chunk
            vad_[i] = torch.tensor(vad)

        return index_, chunk_, vad_

    def tearDown(self):
        clean_tmp_dir(tmp_dir=TMP_DIR)

    def testLoadData(self):

        eq_(self.vad_data.n_seqs, 4)
        eq_(self.vad_data.n_chunks, 9)
        eq_(self.vad_data.size_vad_chunk, 1)
        eq_(self.vad_data.size_vad, torch.Size((9, 1, 2)))

    def testFeed(self):

        to_feed_0 = [
            (0, 0, [[2.3, -5]]),
            (0, 1, [[0.8, 0]]),
            (1, 2, [[1.9, 21]]),
            (3, 0, [[4.2, 3.3]]),
        ]

        index_0, chunk_0, vad_0 = self.build_feeding_vector(to_feed_0)
        self.vad_data.feed_seq_data(index_0, chunk_0, vad_0)

        to_feed_1 = [(1, 0, [[2.0, -1]]), (3, 1, [[0.2, 0.3]])]

        index_1, chunk_1, vad_1 = self.build_feeding_vector(to_feed_1)
        self.vad_data.feed_seq_data(index_1, chunk_1, vad_1)

        expected_vad_0 = torch.tensor([-5, 0])
        print(self.vad_data.get_vad(0, dim=1))
        eq_((expected_vad_0 - self.vad_data.get_vad(0)).norm(), 0)

        expected_vad_1 = torch.tensor([2.0, 0.0, 1.9])
        eq_((expected_vad_1 - self.vad_data.get_vad(1, dim=0)).norm(), 0)

        expected_vad_2 = torch.tensor([0.0, 0.0])
        eq_((expected_vad_2 - self.vad_data.get_vad(2, dim=0)).norm(), 0)

        expected_vad_3 = torch.tensor([3.3, 0.3])
        eq_((expected_vad_3 - self.vad_data.get_vad(3, dim=1)).norm(), 0)


class TestVADParsing(unittest.TestCase):
    def testSquash(self):

        test_vad = [
            0.0,
            0.0,
            0.2,
            0.7,
            0.1,
            0.8,
            0.6,  # 6
            1.0,
            0.0,
            0.0,
            0.0,
            0.7,
            0.0,
            0.0,
            0.0,  # 14
            0.8,
            0.8,
            0.8,
            0.0,
            0.9,
            0.9,
            0.0,
        ]  # 21

        onset = 0.65
        offset = 0.5
        size_step = 0.1
        pad_start = 0.1
        pad_end = 0.1
        offset_start = 0.03

        # 3 -> 0.3 + 0.03 - 0.1 -> 0.23
        # 8 -> 0.8 + 0.03 + 0.1 -> 0.93
        # 12 -> 1.13
        out = get_intervals_from_proba(
            torch.tensor(test_vad),
            size_step,
            offset_start,
            onset,
            offset,
            pad_start,
            pad_end,
        )

        expected_out = [[0.23, 0.93], [1.03, 1.33], [1.43, 2.23]]
        eq_(len(out), len(expected_out))
        for seg_ref, seg_out in zip(out, expected_out):
            ok_(abs(seg_ref[0] - seg_out[0]) < 1e-4)
            ok_(abs(seg_ref[1] - seg_out[1]) < 1e-4)

    def testRemoveVADPyx(self):

        test_vad = [
            0.0,
            0.0,
            0.2,
            0.7,
            0.1,
            0.8,
            0.6,  # 6
            1.0,
            0.0,
            0.0,
            0.0,
            0.7,
            0.0,
            0.0,
            0.0,  # 14
            0.8,
            0.8,
            0.8,
            0.0,
            0.9,
            0.9,
            0.0,
        ]  # 21

        onset = 0.65
        offset = 0.5
        size_step = 0.1
        pad_start = 0.1
        pad_end = 0.1
        offset_start = 0.03

        # 3 -> 0.3 + 0.03 - 0.1 -> 0.23
        # 8 -> 0.8 + 0.03 + 0.1 -> 0.93
        # 12 -> 1.13
        out = get_intervals_from_proba(
            torch.tensor(test_vad),
            size_step,
            offset_start,
            onset,
            offset,
            pad_start,
            pad_end,
        )

        expected_out = [[0.23, 0.93], [1.03, 1.33], [1.43, 2.23]]
        eq_(len(out), len(expected_out))
        for seg_ref, seg_out in zip(out, expected_out):
            ok_(abs(seg_ref[0] - seg_out[0]) < 1e-4)
            ok_(abs(seg_ref[1] - seg_out[1]) < 1e-4)

    def testRemoveVAD(self):

        test_input = [
            [0.1, 0.8],
            [1.3, 10],
            [10.1, 12.0],
            [12.5, 12.8],
            [12.9, 13.5],
            [14, 16],
        ]
        min_size_vad = 1.0
        out = remove_short_seq(test_input, min_size_vad)

        expected_out = [[1.3, 10], [10.1, 12.0], [14, 16]]
        eq_(len(out), len(expected_out))
        for seg_ref, seg_out in zip(out, expected_out):
            ok_(abs(seg_ref[0] - seg_out[0]) < 1e-4)
            ok_(abs(seg_ref[1] - seg_out[1]) < 1e-4)

    def testRemoveVADPyx(self):

        test_input = [
            [0.1, 0.8],
            [1.3, 10],
            [10.1, 12.0],
            [12.5, 12.8],
            [12.9, 13.5],
            [14, 16],
        ]
        min_size_vad = 1.0
        out = remove_short_seq(test_input, min_size_vad)

        expected_out = [[1.3, 10], [10.1, 12.0], [14, 16]]
        eq_(len(out), len(expected_out))
        for seg_ref, seg_out in zip(out, expected_out):
            ok_(abs(seg_ref[0] - seg_out[0]) < 1e-4)
            ok_(abs(seg_ref[1] - seg_out[1]) < 1e-4)

    def testRemoveSILs(self):

        test_input = [
            [0.1, 0.8],
            [1.3, 10],
            [10.1, 12.0],
            [13.5, 13.8],
            [13.9, 14.8],
            [19, 20],
        ]
        min_size_sil = 1.0
        out = merge_short_voids(test_input, min_size_sil)
        expected_out = [[0.0, 12.0], [13.5, 14.8], [19, 20]]
        eq_(len(out), len(expected_out))
        for seg_ref, seg_out in zip(out, expected_out):
            ok_(abs(seg_ref[0] - seg_out[0]) < 1e-4)
            ok_(abs(seg_ref[1] - seg_out[1]) < 1e-4)

    def testRemoveSILsPyx(self):

        test_input = [
            [0.1, 0.8],
            [1.3, 10],
            [10.1, 12.0],
            [13.5, 13.8],
            [13.9, 14.8],
            [19, 20],
        ]
        min_size_sil = 1.0
        out = merge_short_voids(test_input, min_size_sil)
        expected_out = [[0.0, 12.0], [13.5, 14.8], [19, 20]]
        eq_(len(out), len(expected_out))
        for seg_ref, seg_out in zip(out, expected_out):
            ok_(abs(seg_ref[0] - seg_out[0]) < 1e-4)
            ok_(abs(seg_ref[1] - seg_out[1]) < 1e-4)


class TestSeqNameDataset(unittest.TestCase):
    def setUp(self):

        seqs = [
            (
                "0",
                [0.0, 0.9, -1.0, 0.0, 0.0, 0.3, -0.3, 0.4, 0.9, -0.9, 0.5, -0.5],
            ),  # 12 frames
            ("1", [0.6, -1.0, -1.0, 0.5, 0.0, 0.4, -0.4, -0.4, -0.9, 0.9]),  # 10 frames
            ("2", [0.6, 0.1, -1.0, 0.2, 0.4, 0.1, -0.3]),  # 7 frames
            (
                "3",
                [-0.5, 0.5, -0.2, 0.4, -0.4, 0.3, -0.3, 0.8, 0.99, -0.09],
            ),  # 10 frames
            ("4", [-0.1, 0.9, -0.3, 0.3, 0.0, 0.3, -0.3, 0.4]),  # 8 frames
            ("5", [0.0, 0.9, -1.0, 0.0, 0.0, 0.3, -0.3, 0.4, -0.9]),
        ]  # 9 frames

        save_tmp_seq_list(seqs, tmp_dir=TMP_DIR)
        self.seq_names = [f"{x}.wav" for x, _ in seqs]
        self.path_db = TMP_DIR
        self.size_frame = 4
        self.size_batch = 2
        self.dataset = SeqNameDataset(
            self.path_db,
            self.seq_names,
            self.size_frame,
            self.size_batch,
            n_process_loader=10,
            MAX_SIZE_LOADED=28,
        )

    def testLoading(self):

        eq_(self.dataset.n_seqs, 6)
        eq_(self.dataset.n_reloads, 2)
        eq_(len(self.dataset), 6)

    def test_iteration(self):

        expected_batches = torch.tensor(
            [
                [[0.0, 0.9, -1.0, 0.0], [0.0, 0.3, -0.3, 0.4]],
                [[0.9, -0.9, 0.5, -0.5], [0.6, -1.0, -1.0, 0.5]],
                [[0.0, 0.4, -0.4, -0.4], [0.6, 0.1, -1.0, 0.2]],
                [[-0.5, 0.5, -0.2, 0.4], [-0.4, 0.3, -0.3, 0.8]],
                [[-0.1, 0.9, -0.3, 0.3], [0.0, 0.3, -0.3, 0.4]],
                [[0.0, 0.9, -1.0, 0.0], [0.0, 0.3, -0.3, 0.4]],
            ]
        )

        expected_seq_indexes = torch.tensor(
            [[0, 0], [0, 1], [1, 2], [3, 3], [4, 4], [5, 5]], dtype=torch.long
        )

        expected_chunk_indexes = torch.tensor(
            [[0, 1], [2, 0], [1, 0], [0, 1], [0, 1], [0, 1]], dtype=torch.long
        )

        i = 0
        for data, seq_index, chunk_index in iter(self.dataset):
            eq_(data.size(), torch.Size((2, 4, 1)))
            ok_((data.view(2, 4) - expected_batches[i]).norm() < 1e-4)
            eq_((seq_index - expected_seq_indexes[i]).abs().max(), 0)
            eq_((chunk_index - expected_chunk_indexes[i]).abs().max(), 0)
            i += 1
