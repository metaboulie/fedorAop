import os

import numpy
import numpy as np
import torch
from imblearn.metrics import geometric_mean_score
from sklearn.metrics import f1_score, precision_score, recall_score
from torch import nn

from fedorAop.config import N_STEPS_TO_PRINT
from fedorAop.models.sampling import feature_label_split, Sample, Bootstrap


def find_highest_logits_for_columns(logits: torch.Tensor, k: int = 10):
    """For each column, find the k rows where the value is k-largest"""
    pass


def calculate_cost_matrix(labels: numpy.ndarray) -> torch.Tensor:
    """
    Calculates the cost matrix based on the given labels.

    Args:
        labels (numpy.ndarray): Array of labels.

    Returns:
        torch.Tensor: The calculated cost matrix.
    """
    # Count the unique labels and their occurrences
    unique_labels, label_counts = np.unique(labels, return_counts=True)
    num_classes = len(unique_labels)

    # Initialize the cost matrix with all ones
    cost_matrix = np.ones((num_classes, num_classes), dtype=np.float32)

    # Calculate the cost for each pair of classes
    for i in range(num_classes):
        for j in range(num_classes):
            if i == j:
                # Set the cost to 0 for the same class
                cost_matrix[i, j] = 0
            else:
                if label_counts[i] > label_counts[j]:
                    # Calculate the cost based on the label occurrences
                    cost_matrix[i, j] = label_counts[i] / label_counts[j]

    # Convert the cost matrix to a torch tensor
    cost_matrix = torch.tensor(cost_matrix, dtype=torch.float32)

    return cost_matrix


def predict_min_expected_cost_class(cost_matrix: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
    """
    Predicts the class with the minimum expected cost based on the given cost matrix and logits.

    Args:
        cost_matrix (torch.Tensor): A tensor representing the cost matrix.
        logits (torch.Tensor): A tensor representing the logits.

    Returns:
        torch.Tensor: A tensor representing the class with the minimum expected cost.
    """
    # Calculate the expected costs
    expected_costs = torch.matmul(cost_matrix, logits.T).T

    # # Print the first 10 cost values
    # print(f"\nFirst 10th cost: \n{expected_costs[:10]}\n")

    # Find the class with the minimum expected cost
    min_expected_cost_class = torch.argmin(expected_costs, dim=1)

    return min_expected_cost_class


def evaluate(
    data: np.ndarray, model: nn.Module, loss_fn, mode: str = "step", cost_matrix: np.ndarray = None
) -> dict | None:
    """Evaluate the performance of the model on test-set or train-set

    Parameters
    ----------
    data : np.ndarray
        The data
    model : nn.Module
        The trained Neural Network
    loss_fn : _type_
        The loss function of the trained Neural Network
    mode : str, optional
        Options are: ["step", "train", "test", "model"]
        Defaults to "step".
    cost_matrix : np.ndarray, optional
        The cost matrix.
    Returns
    -------
    dict | float | None
        Calculated metrics, depending on the mode
    """
    # Split features and labels
    X, y = feature_label_split(data)

    # Set the model to evaluation mode
    model.eval()
    with torch.no_grad():
        pred = model(X)
        print(f"First 10th logits{pred[:10]}")
        _pred = predict_min_expected_cost_class(cost_matrix, pred)
        # _pred = torch.argmin(pred, dim=1)
        print("Predicted class with the smallest expected cost:", torch.bincount(_pred))
        print(f"First 10th predictions{_pred[:10]}")
        # The geometric mean of accuracy for each label
        correct = geometric_mean_score(_pred, y)
        loss = loss_fn(pred, y).item()

    match mode:
        # Return metrics for 'step' mode
        case "step":
            print(f"G-Mean Accuracy: {(100 * correct): >0.1f}%, Loss: {loss: >5f} \n")
            pass

        # Return metrics for 'train' mode
        case "train":
            print(f"Train Error: \n G-Mean Accuracy: {(100 * correct): >0.1f}%, Loss: {loss: >5f} \n")

        # Return metrics for 'test' mode
        case "test":
            print(f"Test Error: \n G-Mean Accuracy: {(100 * correct): >0.1f}%, Loss: {loss: >5f} \n")

        # Return metrics for 'model' mode
        case "model":
            f1_weighted = f1_score(y, _pred, average="weighted")
            precision_weighted = precision_score(y, _pred, average="weighted")
            recall_weighted = recall_score(y, _pred, average="weighted")

            return {
                "g-mean_accuracy": correct,
                "f1_weighted": f1_weighted,
                "precision_weighted": precision_weighted,
                "recall_weighted": recall_weighted,
            }

        # Raise ValueError for invalid mode
        case _:
            raise ValueError(
                "Please enter a solid value for parameter `mode`, options are ('step', 'train', 'test', 'model')"
            )


def early_stopping(loss_list: list[float], patience: int = 10, threshold: float = 1e-4) -> bool:
    """
    Check if the given list of losses has converged based on a threshold and patience value.

    Args:
        loss_list (list): A list of losses.
        patience (int, optional): The number of previous losses to consider for convergence. Defaults to 10.
        threshold (float, optional): The threshold value for convergence. Defaults to 1e-4.

    Returns:
        bool: True if the losses have converged, False otherwise.
    """
    # Check if there are enough losses to check convergence
    if len(loss_list) < patience + 1:
        print("Not enough losses to check convergence")
        return False

    # Calculate the differences between consecutive losses
    diff = np.abs(np.diff(loss_list)[-patience:])

    # Check if all differences are below the threshold
    if np.all(diff <= threshold):
        print("Losses have converged")
        return True

    print("Losses have not converged")
    return False


def count_unique_labels(data: np.ndarray) -> int:
    """Count the number of unique labels

    Parameters
    ----------
    data : np.ndarray
        The dataset to count unique labels

    Returns
    -------
    int
        The number of unique labels
    """
    # If data is empty, return 0
    if len(data) == 0:
        print("Data is empty")
        return 0

    num_unique_labels = len(np.unique(data[:, -1]))

    # If the last column of data has NA values, subtract 1 from the total count
    if np.isnan(data[:, -1]).any():
        num_unique_labels -= 1
        print("NA values found in the last column")

    print(f"Number of unique labels: {num_unique_labels}")

    return num_unique_labels


def does_model_exist(directory: str, dataset_name: str) -> tuple[bool, str]:
    """
    Check if a model file exists in the specified directory.

    Parameters:
        directory (str): The directory where the model file is located.
        dataset_name (str): The name of the dataset used to train the model.

    Returns:
        tuple[bool, str]: A tuple containing a boolean value indicating whether the model file exists and the path to the model file.
    """
    # Remove the "_train" suffix from dataset_name
    model_name = dataset_name.replace("_train", "") + ".pt"

    # Construct the full path to the model file
    model_path = os.path.join(directory, model_name)

    # Print the model path for debugging
    print(f"Model path: {model_path}")

    # Check if the model file exists
    model_exists = os.path.exists(model_path)

    # Print whether the model file exists or not for debugging
    print(f"Model exists: {model_exists}")

    return model_exists, model_path
