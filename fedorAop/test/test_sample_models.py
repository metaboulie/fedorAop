import numpy as np
import torch
import unittest
from fedorAop.models.sample_models import feature_label_split


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
