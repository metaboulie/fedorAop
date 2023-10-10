import numpy as np
import pytest
from fedorAop.function import early_stopping, count_unique_labels


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
