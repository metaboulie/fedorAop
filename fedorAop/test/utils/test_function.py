from fedorAop.utils.function import (
    early_stopping,
    count_unique_labels,
    calculate_cost_matrix,
    predict_min_expected_cost_class,
)


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


def test_calculate_cost_matrix():
    # # Test case 1: Labels with only one class
    # labels = np.array([0, 0, 0, 0])
    # expected_output = np.array([[0]])
    # assert np.array_equal(calculate_cost_matrix(labels), expected_output)

    # Test case 2: Labels with two classes
    labels = np.array([0, 0, 1, 1])
    expected_output = np.array([[0, 2], [0.5, 0]])
    assert np.array_equal(calculate_cost_matrix(labels), expected_output)

    # Test case 3: Labels with three classes
    labels = np.array([0, 1, 1, 2, 2, 2])
    expected_output = np.array([[0, 1, 1], [2, 0, 0.5], [2, 0.5, 0]])
    assert np.array_equal(calculate_cost_matrix(labels), expected_output)

    # Test case 4: Labels with four classes
    labels = np.array([0, 1, 2, 2, 3, 3, 3, 3])
    expected_output = np.array([[0, 2, 2, 4], [0.5, 0, 1, 2], [0.5, 1, 0, 2], [0.25, 0.5, 0.5, 0]])
    assert np.array_equal(calculate_cost_matrix(labels), expected_output)

    print("All test cases pass")


import numpy as np
import unittest


class TestPredictMinExpectedCostClass(unittest.TestCase):
    def test_min_expected_cost_class(self):
        # Test case where the expected costs are [0.5, 0.2, 0.8, 0.4]
        # The minimum expected cost class is 1 (index starts from 0)
        cost_matrix = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
        prediction_probabilities = np.array([0.1, 0.4, 0.2, 0.3])
        expected_min_class = 1

        result = predict_min_expected_cost_class(cost_matrix, prediction_probabilities)
        self.assertEqual(result, expected_min_class)

    def test_empty_cost_matrix(self):
        # Test case where the cost matrix is empty
        # The function should raise a ValueError
        cost_matrix = np.array([])
        prediction_probabilities = np.array([0.1, 0.4, 0.2, 0.3])

        with self.assertRaises(ValueError):
            predict_min_expected_cost_class(cost_matrix, prediction_probabilities)

    def test_empty_prediction_probabilities(self):
        # Test case where the prediction probabilities are empty
        # The function should raise a ValueError
        cost_matrix = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
        prediction_probabilities = np.array([])

        with self.assertRaises(ValueError):
            predict_min_expected_cost_class(cost_matrix, prediction_probabilities)


if __name__ == "__main__":
    test_early_stopping()
    test_count_unique_labels()
    test_calculate_cost_matrix()
    unittest.main()
