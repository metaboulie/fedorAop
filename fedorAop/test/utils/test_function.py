import unittest

import numpy as np
import torch

from fedorAop.utils.function import (
    early_stopping,
    count_unique_labels,
    calculate_cost_matrix,
    predict_min_expected_cost_class,
    weighted_macro_f1,
    calculate_class_weights,
)


class TestCalculateClassWeights(unittest.TestCase):
    def test_single_class(self):
        labels = np.array([0, 0, 0, 0])
        expected_weights = torch.tensor([1.0])
        weights = calculate_class_weights(labels)
        self.assertTrue(torch.all(torch.eq(weights, expected_weights)))

    def test_multiple_classes(self):
        labels = np.array([0, 0, 1, 1, 2, 2, 2])
        expected_weights = torch.tensor([1.5, 1.5, 1.0])
        weights = calculate_class_weights(labels)
        self.assertTrue(torch.all(torch.eq(weights, expected_weights)))

    def test_empty_labels(self):
        labels = np.array([])
        expected_weights = torch.tensor([])
        weights = calculate_class_weights(labels)
        self.assertTrue(torch.all(torch.eq(weights, expected_weights)))


class TestWeightedMacroF1(unittest.TestCase):
    def test_weighted_macro_f1(self):
        # Test case 1: Two classes with equal number of instances
        y_true = torch.tensor([0, 1, 0, 1])
        y_pred = torch.tensor([0, 0, 1, 1])
        expected_wmf1 = torch.tensor(0.5)
        self.assertEqual(weighted_macro_f1(y_true, y_pred), expected_wmf1)

        # Test case 2: Three classes with unequal number of instances
        y_true = torch.tensor([0, 1, 2, 0, 1, 1])
        y_pred = torch.tensor([0, 1, 2, 1, 0, 1])
        expected_wmf1 = torch.tensor(0.75)
        self.assertAlmostEqual(weighted_macro_f1(y_true, y_pred).item(), expected_wmf1.item())


def test_early_stopping():
    # Test case 1: Losses have converged
    loss_list = [
        0.001,
        0.001,
        0.001,
        0.001,
        0.001,
        0.001,
        0.001,
        0.001,
        0.001,
        0.001,
        0.001,
    ]
    assert early_stopping(loss_list) == True

    # Test case 2: Losses have not converged
    loss_list = [0.01, 0.02, 0.03, 0.02, 0.01, 0.005, 0.002, 0.001, 0.001, 0.01, 0.01]
    assert early_stopping(loss_list) == False

    # Test case 3: Not enough losses to check convergence
    loss_list = [0.01, 0.02, 0.03, 0.02, 0.01]
    assert early_stopping(loss_list) == False

    # Test case 4: Losses have converged with custom patience and threshold
    loss_list = [0.01, 0.02, 0.03, 0.02, 0.003, 0.002, 0.001, 0.001, 0.001, 0.001]
    assert early_stopping(loss_list, patience=5, threshold=0.001) == True

    # Test case 5: Losses have not converged with custom patience and threshold
    loss_list = [0.01, 0.02, 0.03, 0.02, 0.01, 0.005, 0.002, 0.001, 0.001, 0.001]
    assert early_stopping(loss_list, patience=5, threshold=0.0001) == False


def test_count_unique_labels():
    # Test with an empty dataset
    data = np.array([])
    assert count_unique_labels(data) == 0

    # Test with a dataset containing only one label
    data = np.array([[1]])
    assert count_unique_labels(data) == 1

    # Test with a dataset containing multiple unique labels
    data = np.array([[1], [2], [3]])
    assert count_unique_labels(data) == 3

    # Test with a dataset containing duplicate labels
    data = np.array([[1], [2], [2], [3], [3], [3]])
    assert count_unique_labels(data) == 3

    # Test with a dataset containing multiple columns
    data = np.array([[1, 2], [2, 3], [3, 4]])
    assert count_unique_labels(data) == 3

    # Test with a dataset containing NA values
    data = np.array([[1, 2], [2, 3], [4, np.nan]])
    assert count_unique_labels(data) == 2


class TestCalculateCostMatrix(unittest.TestCase):
    def test_same_class(self):
        labels = np.array([0, 0, 0, 0])
        expected_cost_matrix = torch.tensor([[0]], dtype=torch.float32)
        cost_matrix = calculate_cost_matrix(labels)
        self.assertTrue(torch.all(torch.eq(cost_matrix, expected_cost_matrix)))

    def test_different_classes(self):
        labels = np.array([0, 1, 2, 3])
        expected_cost_matrix = torch.tensor(
            [[0, 1, 1, 1], [1, 0, 1, 1], [1, 1, 0, 1], [1, 1, 1, 0]],
            dtype=torch.float32,
        )
        cost_matrix = calculate_cost_matrix(labels)
        self.assertTrue(torch.all(torch.eq(cost_matrix, expected_cost_matrix)))

    def test_multiple_occurrences(self):
        labels = np.array([0, 0, 1, 1, 1, 2, 2, 2, 2])
        expected_cost_matrix = torch.tensor([[0, 1, 1], [3 / 2, 0, 1], [2, 4 / 3, 0]], dtype=torch.float32)
        cost_matrix = calculate_cost_matrix(labels)
        self.assertTrue(torch.all(torch.eq(cost_matrix, expected_cost_matrix)))


class TestPredictMinExpectedCostClass(unittest.TestCase):
    def setUp(self):
        # Create test inputs
        self.cost_matrix = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=torch.float32)
        self.logits = torch.tensor([[0.1, 0.2, 0.7], [0.3, 0.4, 0.3], [0.6, 0.2, 0.2]])

    def test_predict_min_expected_cost_class(self):
        # Call the function
        result = predict_min_expected_cost_class(self.cost_matrix, self.logits)

        # Check the output shape
        self.assertEqual(result.shape, (3,))
        # Check the output values
        self.assertEqual(result.tolist(), [0, 0, 0])


if __name__ == "__main__":
    test_early_stopping()
    test_count_unique_labels()
    unittest.main()
