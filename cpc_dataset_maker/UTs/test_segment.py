import unittest

from cpc_dataset_maker.transforms.segmentation import update_timeline_from_segmentation

#####################################
# UTILS


#####################################


class TestSegmentTimeline(unittest.TestCase):

    # Assert list equal fails with some floating values
    def check_timelines_equal(self, t1, t2):

        self.assertEqual(len(t1), len(t2))
        n_subs = len(t1)

        for sub in range(n_subs):
            self.assertEqual(len(t1[sub]), len(t2[sub]))
            size_sub = len(t1[sub])

            for i in range(size_sub):
                self.assertTrue(len(t1[sub][i]) == 2)
                self.assertTrue(len(t2[sub][i]) == 2)
                self.assertTrue(abs(t1[sub][i][0] - t2[sub][i][0]) < 1e-4)
                self.assertTrue(abs(t1[sub][i][1] - t2[sub][i][1]) < 1e-4)

    def test_split_segment(self):

        pipeline = [(1.5, 2.0), (2.1, 2.3), (10.0, 20.0), (30.0, 41.0)]
        segmentation = [1.0, 1.1, 1.3, 3.0, 25.0, 31.0, 33.0, 42.0]

        new_pipeline = update_timeline_from_segmentation(pipeline, segmentation)
        expected_pipeline = [
            [],
            [],
            [],
            [(0.2, 0.7), (0.8, 1.0)],
            [(7.0, 17.0)],
            [(5.0, 6.0)],
            [(0.0, 2.0)],
            [(0.0, 8.0)],
            [],
        ]

        self.assertTrue(len(new_pipeline), len(segmentation) + 1)
        self.check_timelines_equal(new_pipeline, expected_pipeline)

    def test_no_split(self):

        pipeline = [(1.5, 2.0), (2.1, 2.3), (10.0, 20.0), (30.0, 41.0)]
        seg1 = []

        new_pipeline = update_timeline_from_segmentation(pipeline, seg1)
        self.check_timelines_equal(new_pipeline, [pipeline])

        seg1 = [0.1, 0.2, 0.3, 0.4, 1.0, 45.0, 70.0]
        new_pipeline = update_timeline_from_segmentation(pipeline, seg1)

        expected_pipeline = [
            [],
            [],
            [],
            [],
            [],
            [(0.5, 1.0), (1.1, 1.3), (9.0, 19.0), (29.0, 40.0)],
            [],
            [],
        ]
        self.check_timelines_equal(new_pipeline, expected_pipeline)

    def test_size_split(self):

        pipeline = [(1.5, 2.0), (2.1, 2.3), (10.0, 20.0), (30.0, 41.0)]
        segmentation = [1, 2, 3, 4, 5, 6]
        new_pipeline = update_timeline_from_segmentation(pipeline, segmentation)
        self.assertTrue(len(new_pipeline), len(segmentation) + 1)

    def test_final_split(self):

        pipeline = [(1.5, 2.0), (2.1, 2.3), (10.0, 20.0), (30.0, 41.0)]
        segmentation = [35.0]
        new_pipeline = update_timeline_from_segmentation(pipeline, segmentation)
        self.assertTrue(len(new_pipeline), 2)

        expected_pipeline = [
            [(1.5, 2.0), (2.1, 2.3), (10.0, 20.0), (30.0, 35.0)],
            [(0.0, 6.0)],
        ]
        self.check_timelines_equal(new_pipeline, expected_pipeline)

    def test_single_split(self):

        pipeline = [(0.03, 12.3), (12.4, 70)]
        segmentation = [10.0, 20.0, 30.0]
        new_pipeline = update_timeline_from_segmentation(pipeline, segmentation)
        self.assertTrue(len(new_pipeline), 2)

        expected_pipeline = [
            [(0.03, 10.0)],
            [(0, 2.3), (2.4, 10.0)],
            [(0.0, 10.0)],
            [(0.0, 40.0)],
        ]
        self.check_timelines_equal(new_pipeline, expected_pipeline)
