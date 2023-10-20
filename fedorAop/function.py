import os

import torch
from sklearn.metrics import f1_score, precision_score, recall_score
from torch import nn
import numpy as np
from fedorAop.config import N_STEPS_TO_PRINT
from fedorAop.sample import feature_label_split, Sample, Bootstrap
from imblearn.metrics import geometric_mean_score


def train_loop(
    data: np.ndarray,
    model: nn.Module,
    loss_fn,
    optimizer,
    count: int,
    sample_model: Sample = Bootstrap,
):
    """
    Trains the model during every step.

    Parameters:
    - data: np.ndarray
        The dataset to train.
    - model: nn.Module
        The Neural Network.
    - loss_fn: _type_
        The loss function of the Neural Network.
    - optimizer: _type_
        The optimizer to perform gradient descent.
    - count: int
        The count of the current step.
    - sample_model: Sample, optional
        The model to resample the dataset, see sample.py for more details.
        Defaults to Bootstrap.
    """

    # Utilize the sampling-model to resample the dataset
    X_train, y_train = sample_model.sample

    # Clear the gradients
    optimizer.zero_grad()

    # Get the predictions of the neural network
    pred = model(X_train)

    # Calculate the loss
    _loss_fn = loss_fn(pred, y_train)

    # Calculate gradients
    _loss_fn.backward()

    # Perform gradient descent
    optimizer.step()

    # Print result every M steps
    if count % N_STEPS_TO_PRINT == 0:
        evaluate(data=data, model=model, loss_fn=_loss_fn, mode="step")


def evaluate(
    data: np.ndarray, model: nn.Module, loss_fn, mode: str = "step"
) -> dict | float | None:
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
        # The geometric mean of accuracy for each label
        correct = geometric_mean_score(pred.argmax(1), y)
        loss = loss_fn(pred, y).item()

    match mode:
        # Return metrics for 'train' mode
        case "train":
            print(
                f"Train Error: \n G-Mean Accuracy: {(100 * correct): >0.1f}%, Loss: {loss: >5f} \n"
            )
            return loss
        case "step":
            print(f"G-Mean Accuracy: {(100 * correct): >0.1f}%, Loss: {loss: >5f} \n")
            pass

        # Return metrics for 'step' mode
        case "train":
            print(
                f"Train Error: \n G-Mean Accuracy: {(100 * correct): >0.1f}%, Loss: {loss: >5f} \n"
            )
            return loss

        # Return metrics for 'test' mode
        case "test":
            print(
                f"Test Error: \n G-Mean Accuracy: {(100 * correct): >0.1f}%, Loss: {loss: >5f} \n"
            )
            return loss

        # Return metrics for 'model' mode
        case "model":
            _pred = pred.argmax(1)

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


def early_stopping(
    loss_list: list[float], patience: int = 10, threshold: float = 1e-4
) -> bool:
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
