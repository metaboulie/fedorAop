import unittest
from unittest.mock import patch
from fedorAop.io import (
    read_h5ad_datasets,
    convert_dataset_dict_to_np_dict,
)
import anndata as ad
import numpy as np
from deepdiff import DeepDiff


class TestReadH5adDatasets(unittest.TestCase):
    @patch("os.walk")
    @patch("anndata.read_h5ad")
    def test_read_h5ad_datasets(self, mock_read_h5ad, mock_walk):
        # Test case 1: Empty directory
        mock_walk.return_value = []
        result = read_h5ad_datasets("path/to/directory")
        self.assertEqual(result, {})

        # Test case 2: Single dataset in directory
        mock_walk.return_value = [("path/to/directory", [], ["dataset.h5ad"])]
        mock_read_h5ad.return_value = "data"
        result = read_h5ad_datasets("path/to/directory")
        self.assertEqual(result, {"dataset": "data"})

        # Test case 3: Multiple datasets in directory
        mock_walk.return_value = [
            ("path/to/directory", [], ["dataset1.h5ad", "dataset2.h5ad"])
        ]
        mock_read_h5ad.side_effect = ["data1", "data2"]
        result = read_h5ad_datasets("path/to/directory")
        self.assertEqual(result, {"dataset1": "data1", "dataset2": "data2"})

        # Test case 4: Subdirectories in directory
        mock_walk.return_value = [
            ("path/to/directory", ["subdir"], []),
            ("path/to/directory/subdir", [], ["dataset.h5ad"]),
        ]
        mock_read_h5ad.side_effect = ["data"]
        result = read_h5ad_datasets("path/to/directory")
        self.assertEqual(result, {"dataset": "data"})

        # Test case 5: Multiple subdirectories in directory
        mock_walk.return_value = [
            ("path/to/directory", ["subdir1", "subdir2"], []),
            ("path/to/directory/subdir1", [], ["dataset1.h5ad"]),
            ("path/to/directory/subdir2", [], ["dataset2.h5ad"]),
        ]
        mock_read_h5ad.side_effect = ["data1", "data2"]
        result = read_h5ad_datasets("path/to/directory")
        self.assertEqual(result, {"dataset1": "data1", "dataset2": "data2"})


class TestConvertDatasetDictToNpDict(unittest.TestCase):
    def test_empty_dataset_dict(self):
        dataset_dict = {}
        expected_output = {}
        self.assertEqual(convert_dataset_dict_to_np_dict(dataset_dict), expected_output)

    def test_single_dataset(self):
        dataset_dict = {
            "dataset1": ad.AnnData(
                X=np.array([[1, 2], [3, 4]]), obs={"cell.type": ["A", "B"]}
            )
        }
        expected_output = {"dataset1": np.array([[1, 2, 0], [3, 4, 1]])}
        self.assertEqual(
            DeepDiff(convert_dataset_dict_to_np_dict(dataset_dict), expected_output), {}
        )

    def test_multiple_datasets(self):
        dataset_dict = {
            "dataset1": ad.AnnData(
                X=np.array([[1, 2], [3, 4]]), obs={"cell.type": ["A", "B"]}
            ),
            "dataset2": ad.AnnData(
                X=np.array([[5, 6], [7, 8]]), obs={"cell.type": ["C", "D"]}
            ),
        }
        expected_output = {
            "dataset1": np.array([[1, 2, 0], [3, 4, 1]]),
            "dataset2": np.array([[5, 6, 0], [7, 8, 1]]),
        }
        self.assertEqual(
            DeepDiff(convert_dataset_dict_to_np_dict(dataset_dict), expected_output), {}
        )


if __name__ == "__main__":
    unittest.main()
