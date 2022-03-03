import unittest

from numpy import dtype
import torch
from cpc_dataset_maker.transforms.extend_silences import (
    update_speech_activity_from_new_silence,
    add_silences_to_speech_mono,
    make_ramp,
    merge_sils,
)


class TestPipelineExtension(unittest.TestCase):
    def test_update_pipeline(self):

        ref_speech_activity = [(0.0, 1.0), (1.5, 10.0), (11.0, 12.0)]
        ref_silences = [(0.5, 0.3), (1.1, 0.2), (1.5, 2), (9.0, 1.0), (13.0, 1.0)]

        new_timeline = update_speech_activity_from_new_silence(
            ref_speech_activity, ref_silences
        )

        expected_timeline = [
            (0.0, 0.5),
            (0.8, 1.3),
            (4.0, 11.5),
            (12.5, 13.5),
            (14.5, 15.5),
        ]
        self.assertListEqual(new_timeline, expected_timeline)

    def test_update_before_end(self):
        ref_speech_activity = [(0.0, 1.0), (1.5, 10.0), (11.0, 12.0)]
        ref_silences = [(2.0, 3.0), (4.0, 7.0), (11.0, 12.0)]

        new_timeline = update_speech_activity_from_new_silence(
            ref_speech_activity, ref_silences
        )
        expected_timeline = [
            (0.0, 1.0),
            (1.5, 2.0),
            (5.0, 7.0),
            (14.0, 20.0),
            (33.0, 34.0),
        ]
        self.assertListEqual(new_timeline, expected_timeline)

    def test_multiple_silences_in_vad(self):
        ref_speech_activity = [(1.0, 10.0), (12.0, 20.0), (22.0, 23.0)]
        ref_silences = [
            (2.0, 3.0),
            (4.0, 6.0),
            (5.0, 1.0),
            (11.0, 12.0),
            (13.0, 2.0),
            (14.0, 1.0),
        ]

        new_timeline = update_speech_activity_from_new_silence(
            ref_speech_activity, ref_silences
        )
        expected_timeline = [
            (1.0, 2.0),
            (5.0, 7.0),
            (13.0, 14.0),
            (15.0, 20.0),
            (34.0, 35.0),
            (37.0, 38.0),
            (39.0, 45.0),
            (47.0, 48.0),
        ]
        self.assertListEqual(new_timeline, expected_timeline)

    def test_update_after_end(self):
        ref_speech_activity = [(1.0, 10.0), (12.0, 20.0), (22.0, 23.0)]
        ref_silences = [(40.0, 2.0), (45.0, 10.0), (100.0, 112.2123)]
        new_timeline = update_speech_activity_from_new_silence(
            ref_speech_activity, ref_silences
        )

        self.assertListEqual(new_timeline, ref_speech_activity)

    def test_update_before_begining(self):
        ref_speech_activity = [(1.0, 10.0), (12.0, 20.0), (22.0, 23.0)]
        ref_silences = [(0.1, 1), (0.2, 2.5), (0.44, 1.4), (0.542, 0.6), (0.88, 0.5)]
        new_timeline = update_speech_activity_from_new_silence(
            ref_speech_activity, ref_silences
        )

        expected_timeline = [(7.0, 16.0), (18.0, 26.0), (28.0, 29.0)]

        self.assertListEqual(new_timeline, expected_timeline)


class TestRamp(unittest.TestCase):
    def test_unit_increasing_ramp(self):

        ramp10 = make_ramp(torch.ones(10, dtype=torch.float), 0.0, 1.0)
        expected_10 = torch.tensor(
            [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], dtype=torch.float
        )
        self.assertEqual((ramp10 - expected_10).norm(), 0)

        ramp5 = make_ramp(torch.ones(5, dtype=torch.float), 0.0, 1.0)
        expected_5 = torch.tensor([0.0, 0.2, 0.4, 0.6, 0.8], dtype=torch.float)
        self.assertEqual((ramp5 - expected_5).norm(), 0)

    def test_unit_decreasing_ramp(self):

        ramp10 = make_ramp(torch.ones(10, dtype=torch.float), 1.0, 0.0)
        expected_10 = torch.tensor(
            [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1], dtype=torch.float
        )
        self.assertEqual((ramp10 - expected_10).norm(), 0)

        ramp5 = make_ramp(torch.ones(5, dtype=torch.float), 1.0, 0.0)
        expected_5 = torch.tensor([1.0, 0.8, 0.6, 0.4, 0.2], dtype=torch.float)
        self.assertEqual((ramp5 - expected_5).norm(), 0)

    def base_increasing_ramp(self):

        input_data = torch.tensor([10.0, 20.0, 15.0, 32.5, 21.2], dtype=float)
        ramp = make_ramp(input_data, 0.0, 1.0)

        expected_ramp = torch.tensor([0.0, 4.0, 6.0, 19.5, 16.96], dtype=float)
        self.assertTrue((ramp - expected_ramp).norm() < 1e-4)

    def test_empty_ramp(self):

        empty = torch.ones((0, 2), dtype=torch.float)
        ramp_empty = make_ramp(empty, 1.0, 0.0)
        self.assertEqual((empty - ramp_empty).norm(), 0)

    def test_unit_ramp(self):

        unit = torch.randn((1), dtype=torch.float)
        ramp_unit = make_ramp(unit, 1.0, 0.0)
        self.assertEqual((unit - ramp_unit).norm(), 0)


class TestAddSilence(unittest.TestCase):
    def test_add_silence(self):

        ref_audio = torch.tensor(
            [
                1.0,
                2.0,
                1.0,
                2.0,
                1.0,
                3.0,
                2.0,
                1.0,
                -1.0,
                -2.0,
                -3.0,
                -1.0,
                0.0,
                1.0,
                2.0,
                1.0,
                2.0,
                1.0,
                2.0,
                1.0,
                3.0,
                2.0,
                1.0,
                -1.0,
                -2.0,
                -3.0,
                -1.0,
                0.0,
                1.0,
                2.0,
            ]
        )
        crossfade_frame = 5

        silences = [(3, 11), (27, 10)]

        extended_audio = add_silences_to_speech_mono(
            ref_audio, silences, crossfade_frame
        )

        expected_output = torch.tensor(
            [
                1.0,
                2.0,
                1.0,
                2.0,  # 1 * 2
                0.8,  # 0.8 * 1
                1.8,  # 0.6 * 3
                0.8,  # 0.4 * 2
                0.2,  # 0.2 * 1
                0.0,
                0.0,  # 0 * -1
                -0.4,  # 0.2 * -2
                -1.2,  # 0.4 * -3
                -0.6,  # 0.6 * -1
                0.0,  # 0.8 * 0
                1.0,
                2.0,
                1.0,
                2.0,
                1.0,
                2.0,
                1.0,
                3.0,
                2.0,
                1.0,
                -1.0,
                -2.0,
                -3.0,
                -1.0,
                0.0,  # 0.0 * 1
                0.66667,  # 0.66 * 1
                0.66667,  # 0.33 * 2
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ],
            dtype=torch.float,
        )

        self.assertTrue((expected_output - extended_audio).norm() < 1e-4)

    def test_sil_begining(self):

        ref_audio = torch.tensor([1.0, -1.0, 1.0, -1.0, 1.0, -1.0], dtype=torch.float)
        silences = [(-1, 4)]
        crossfade_frame = 3

        extended_audio = add_silences_to_speech_mono(
            ref_audio, silences, crossfade_frame
        )

        expected_output = torch.tensor([0.0, 0.0, -0.3333, 0.66667, -1.0, 1.0, -1.0])
        self.assertTrue((expected_output - extended_audio).norm() < 1e-4)

    def test_sil_begining_and_end(self):

        ref_audio = torch.tensor([1.0, -1.0, 1.0, -1.0, 1.0, -1.0], dtype=torch.float)
        silences = [(-1, 4), (5, 4)]
        crossfade_frame = 3

        extended_audio = add_silences_to_speech_mono(
            ref_audio, silences, crossfade_frame
        )

        expected_output = torch.tensor(
            [0.0, 0.0, -0.3333, 0.66667, -1.0, 1.0, -1.0, 0.0, 0.0, 0.0]
        )
        self.assertTrue((expected_output - extended_audio).norm() < 1e-4)

    def test_fail(self):

        ref_audio = torch.tensor([1.0, -1.0, 1.0, -1.0, 1.0, -1.0], dtype=torch.float)
        has_failed = False
        silences = [(-2, 10), (-3, 2)]
        crossfade_frame = 3
        try:
            add_silences_to_speech_mono(ref_audio, silences, crossfade_frame)
        except RuntimeError:
            has_failed = True

        self.assertTrue(has_failed)

    def test_no_crossfade(self):

        ref_audio = torch.tensor([1.0, -1.0, 1.0, -1.0, 1.0, -1.0], dtype=torch.float)
        silences = [(1, 3), (5, 1)]
        crossfade_frame = 0

        extended_audio = add_silences_to_speech_mono(
            ref_audio, silences, crossfade_frame
        )

        expected_output = torch.tensor([1.0, 0.0, 0.0, 0.0, -1.0, 1.0, -1.0, 1.0, 0.0])

        print(extended_audio)
        self.assertTrue((expected_output - extended_audio).norm() < 1e-4)


class TestMergeSils(unittest.TestCase):
    def test_merge_sils(self):

        sils = [(1, 2), (4, 3), (12, 2), (17, 1), (18, 3)]
        crossfade = 1

        merged = merge_sils(sils, crossfade)
        print(merged)
        expected = [(1, 6), (12, 2), (17, 4)]
        self.assertListEqual(merged, expected)
