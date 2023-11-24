import os

import numpy as np
import torch
from imblearn.metrics import geometric_mean_score
from torch import nn
from torcheval.metrics.functional import multiclass_precision, multiclass_recall
from tqdm import tqdm

from fedorAop.config import *
from fedorAop.models.neural_network import NeuralNetwork, WeightedCrossEntropyLoss
from fedorAop.models.sampling import feature_label_split


def find_highest_logits_for_columns(logits: torch.Tensor, k: int = 10):
    """For each column, find the k rows where the value is k-largest"""
    return NotImplemented


def calculate_class_weights(labels: np.ndarray | torch.Tensor) -> torch.Tensor:
    """
    Calculate the class weights for a given dataset.

    Args:
        labels (np.ndarray | torch.Tensor): The input labels.

    Returns:
        torch.Tensor: The class weights.
    """
    labels = torch.tensor(labels, dtype=torch.int)

    # Count the number of occurrences of each class label
    class_counts = torch.bincount(labels)

    return torch.max(class_counts) / class_counts if len(class_counts) > 0 else torch.tensor([])


def weighted_macro_f1(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.float32:
    """
    Calculate the weighted macro F1-score.

    Args:
        y_true (torch.Tensor): True labels.
        y_pred (torch.Tensor): Predicted labels.

    Returns:
        torch.float32: Weighted macro F1-score.
    """
    # Compute the number of instances for each class
    n_instances = len(y_true)
    n_classes = len(torch.unique(y_true))
    n_instances_per_class = torch.bincount(y_true)

    # Compute the weights for each class
    weights = [(n_instances - n) / ((n_classes - 1) * n_instances) for n in n_instances_per_class]
    weights = torch.tensor(weights, dtype=torch.float32)

    # Compute the class-specific metrics
    ppv = multiclass_precision(y_pred, y_true, average=None, num_classes=n_classes)
    tpr = multiclass_recall(y_pred, y_true, average=None, num_classes=n_classes)

    weighted_ppv = (ppv * weights).sum(dim=0) / weights.sum(dim=0)
    weighted_tpr = (tpr * weights).sum(dim=0) / weights.sum(dim=0)

    wmf1 = 2 * (weighted_ppv * weighted_tpr) / (weighted_ppv + weighted_tpr)

    return wmf1.item()


def calculate_cost_matrix(labels: np.ndarray | torch.Tensor) -> torch.Tensor:
    """
    Calculates the cost matrix based on the given labels.

    Args:
        labels (torch.Tensor): Array of labels.

    Returns:
        torch.Tensor: The calculated cost matrix.
    """
    # Count the unique labels and their occurrences
    labels = torch.tensor(labels, dtype=torch.int)
    unique_labels, label_counts = torch.unique(labels, return_counts=True)
    num_classes = len(unique_labels)

    # Initialize the cost matrix with all ones
    cost_matrix = torch.ones((num_classes, num_classes), dtype=torch.float32)

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

    return torch.argmin(expected_costs, dim=1)


def evaluate(
    data: np.ndarray, model: nn.Module, loss_fn, mode: str = "step", cost_matrix: torch.Tensor = None, **kwargs
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
        logits = model(X)
        y_pred = predict_min_expected_cost_class(cost_matrix, logits)
        # The geometric mean of accuracy for each label
        correct = geometric_mean_score(y_pred, y)
        loss = loss_fn(logits, y).item()

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
            class_weights = calculate_class_weights(data[:, -1])
            weighted_cross_entropy_loss = WeightedCrossEntropyLoss(class_weights).forward(logits, y).item()
            weighted_macro_F1 = weighted_macro_f1(y, y_pred)

            return {
                "g_mean_accuracy": correct,
                "weighted_macro_f1": weighted_macro_F1,
                "weighted_cross_entropy_loss": weighted_cross_entropy_loss,
                **kwargs,
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
        tuple[bool, str]: A tuple containing a boolean value indicating whether
        the model file exists and the path to the model file.
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


def train_phase_one(
    data: np.ndarray,
    test_data: np.ndarray,
    model: NeuralNetwork,
    loss_fn: nn.Module,
    optimizer,
    n_steps: int,
    cost_matrix: torch.Tensor,
    sample_model,
):
    for epoch in tqdm(range(N_EPOCHS)):
        count = 0
        print(f"Epoch {epoch + 1}\n-------------------------------")
        model.train()

        for _ in tqdm(range(n_steps)):
            count += 1
            X_train, y_train = sample_model.sample()
            optimizer.zero_grad()
            pred = model(X_train)
            loss = loss_fn(pred, y_train)
            loss.backward()
            optimizer.step()
            if count % N_STEPS_TO_PRINT == 0:
                evaluate(data=data, model=model, loss_fn=loss_fn, mode="step", cost_matrix=cost_matrix)

        evaluate(data=data, model=model, loss_fn=loss_fn, mode="train", cost_matrix=cost_matrix)
        evaluate(data=test_data, model=model, loss_fn=loss_fn, mode="test", cost_matrix=cost_matrix)


def train_phase_two(
    data: np.ndarray,
    test_data: np.ndarray,
    model: NeuralNetwork,
    loss_fn: nn.Module,
    optimizer,
    n_steps: int,
    cost_matrix: torch.Tensor,
    sample_model,
):
    X, y = data[:, :-1], data[:, -1]
    model.layers[0].frozen()
    active_layers = model.layers[1]
    X = torch.tensor(X, dtype=torch.float32, requires_grad=False)
    y = torch.tensor(y, dtype=torch.long)
    X = model.layers[0].forward(X)
    for epoch in tqdm(range(N_EPOCHS)):
        count = 0
        print(f"Epoch {epoch + 1}\n-------------------------------")
        active_layers.train()

        for _ in tqdm(range(n_steps)):
            count += 1
            X_train, y_train = sample_model.sample()
            optimizer.zero_grad()
            pred = active_layers(X_train)
            loss = loss_fn(pred, y_train)
            loss.backward()
            optimizer.step()
            if count % N_STEPS_TO_PRINT == 0:
                model.layers[1] = active_layers
                evaluate(data=data, model=model, loss_fn=loss_fn, mode="step", cost_matrix=cost_matrix)

        model.layers[1] = active_layers

        evaluate(data=data, model=model, loss_fn=loss_fn, mode="train", cost_matrix=cost_matrix)
        evaluate(data=test_data, model=model, loss_fn=loss_fn, mode="test", cost_matrix=cost_matrix)


def two_phase_train(
    data: np.ndarray,
    test_data: np.ndarray,
    model: NeuralNetwork,
    loss_fn: nn.Module,
    cost_matrix: torch.Tensor,
    sample_model_one,
    sample_model_two,
):
    n_steps = data.shape[0] // BATCH_SIZE
    optimizer = torch.optim.Adam(model.parameters(), ADAM_LR, betas=(ADAM_BETA1, ADAM_BETA2), eps=ADAM_EPS)

    train_phase_one(data, test_data, model, loss_fn, optimizer, n_steps, cost_matrix, sample_model_one)

    optimizer = torch.optim.Adam(model.parameters(), ADAM_LR, betas=(ADAM_BETA1, ADAM_BETA2), eps=ADAM_EPS)
    train_phase_two(data, test_data, model, loss_fn, optimizer, n_steps, cost_matrix, sample_model_two)
