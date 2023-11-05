import unittest

import numpy as np
import scipy
import pandas as pd
import torch

from fedorAop.models.sampling import feature_label_split, average_rank, generate_probabilities


class TestAverageRank(unittest.TestCase):
    def test_average_rank_descending(self):
        # Test case when mode is "descending"
        df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6], "C": [7, 8, 9]})
        columns = ["A", "B"]
        expected_output = pd.DataFrame(
            {"A": [1, 2, 3], "B": [4, 5, 6], "C": [7, 8, 9], "Average Rank": [3.0, 2.0, 1.0]}
        )
        result = average_rank(df, columns, mode="descending")
        pd.testing.assert_frame_equal(result, expected_output)

    def test_average_rank_ascending(self):
        # Test case when mode is "ascending"
        df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6], "C": [7, 8, 9]})
        columns = ["A", "B"]
        expected_output = pd.DataFrame(
            {"A": [1, 2, 3], "B": [4, 5, 6], "C": [7, 8, 9], "Average Rank": [1.0, 2.0, 3.0]}
        )
        result = average_rank(df, columns, mode="ascending")
        pd.testing.assert_frame_equal(result, expected_output)

    def test_average_rank_no_existing_column(self):
        # Test case when the DataFrame doesn't have the "Average Rank" column
        df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6], "C": [7, 8, 9]})
        columns = ["A", "B"]
        expected_output = pd.DataFrame(
            {"A": [1, 2, 3], "B": [4, 5, 6], "C": [7, 8, 9], "Average Rank": [1.0, 2.0, 3.0]}
        )
        result = average_rank(df, columns, mode="ascending")
        pd.testing.assert_frame_equal(result, expected_output)


def test_generate_probabilities():
    # Test case 1: Generate probabilities from the default distribution (scipy.stats.norm)
    size = 10
    probabilities = generate_probabilities(size)
    assert len(probabilities) == size
    assert np.all(probabilities >= 0) and np.all(probabilities <= 1)

    # Test case 2: Generate probabilities from a different distribution (scipy.stats.uniform)
    size = 5
    distribution = scipy.stats.uniform
    probabilities = generate_probabilities(size, distribution)
    assert len(probabilities) == size
    assert np.all(probabilities >= 0) and np.all(probabilities <= 1)

    # Test case 3: Generate probabilities with additional arguments for the distribution
    size = 100
    distribution = scipy.stats.expon
    scale = 2.0
    probabilities = generate_probabilities(size, distribution, scale=scale)
    assert len(probabilities) == size
    assert np.all(probabilities >= 0) and np.all(probabilities <= 1)


class TestFeatureLabelSplit(unittest.TestCase):
    def setUp(self):
        self.data = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])

    def test_data_type(self):
        X, y = feature_label_split(self.data)
        self.assertIsInstance(X, torch.Tensor)
        self.assertIsInstance(y, torch.Tensor)

    def test_data_shape(self):
        X, y = feature_label_split(self.data)
        self.assertEqual(X.shape, (2, 3))
        self.assertEqual(y.shape, (2,))

    def test_data_dtype(self):
        X, y = feature_label_split(self.data)
        self.assertEqual(X.dtype, torch.float32)
        self.assertEqual(y.dtype, torch.long)


if __name__ == "__main__":
    unittest.main()
