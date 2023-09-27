import os

import torch
from typing import List, Tuple
import polars as pl
from sklearn.metrics import f1_score, precision_score, recall_score
from torch import nn
import numpy as np
from config import BATCH_SIZE
from sample import featureLabelSplit, Sample


def trainLoop(
    data: np.ndarray,
    model: nn.Module,
    loss_fn,
    optimizer,
    count: int,
    correct: list,
    loss: list,
    sampleModel: Sample,
):
    """
    How the model is trained during every step.
    """
    # Mini-Batch
    X_train, y_train = sampleModel.sample

    # Clarify the gradients
    optimizer.zero_grad()

    pred = model(X_train)

    # Calculate the loss
    loss_fn = loss_fn(pred, y_train)

    loss.append(loss_fn.item())

    correct.append(np.sum(np.array(pred.argmax(1) == y_train)) / len(y_train))

    # Calculate gradients
    loss_fn.backward()

    # Gradient descent
    optimizer.step()

    if count % 20 == 0:
        correct_, loss_ = sum(correct) / len(correct), sum(loss) / len(loss)
        print(
            f"Accuracy: {(100 * correct_):>0.1f}%, loss: {loss_:>7f}  [{(BATCH_SIZE * count):>5d}/{data.shape[0]:>5d}]"
        )


def evaluateTest(data: np.ndarray, model: nn.Module, loss_fn, test: bool = True) -> str:
    """
    Evaluate the performance of the model on the current stage
    """
    X, y = featureLabelSplit(data)

    # Set the model to evaluation mode
    model.eval()
    with torch.no_grad():
        pred = model(X)
        test_loss = loss_fn(pred, y).item()
        correct = np.sum(np.array(pred.argmax(1) == y)) / len(y)

        if test:
            print(
                f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n"
            )

    return test_loss


def evaluateModel(data: np.ndarray, model: nn.Module) -> list:
    """
    Evaluate the performance of the trained model
    """
    X, y = featureLabelSplit(data)

    model.eval()
    with torch.no_grad():
        pred = model(X)

    pred = pred.argmax(1)

    f1_micro = f1_score(y, pred, average="micro")
    f1_macro = f1_score(y, pred, average="macro")
    precision = precision_score(y, pred, average="weighted")
    recall = recall_score(y, pred, average="weighted")

    return [f1_micro, f1_macro, precision, recall]


def earlyStopping(
    loss_list: list, count: int, patience: int = 10, threshold: float = 1e-4
) -> bool:
    """
    Stop training if no progress is made anymore
    """
    if len(loss_list) < 2:
        return False
    if (loss_list[-1] - loss_list[-2]) ** 2 < threshold:
        count += 1
    else:
        count = 0
    if count > patience:
        return True
    return False


def countUniqueLabels(data: np.ndarray) -> int:
    # Extract the last column of the 2D NumPy array
    labels_column = data[:, -1]

    # Find the unique labels
    unique_labels = np.unique(labels_column)

    return len(unique_labels)


def createLossDataframe(data: List[List], epoch_count: int) -> pl.DataFrame:
    schema = ["Dataset"] + [f"epoch{i}" for i in range(1, epoch_count + 1)]

    return pl.DataFrame(data, schema)


def createMetricsDataframe(data: List[List]) -> pl.DataFrame:
    schema = [
        "Dataset",
        "f1 micro",
        "f1 macro",
        "precision weighted",
        "recall weighted",
    ]

    return pl.DataFrame(data, schema)


def doesModelExist(directory: str, dataset_name: str) -> Tuple[bool, str]:
    # Define the Models folder path
    models_folder = directory

    # Remove the "_train" suffix from dataset_name
    model_name = dataset_name.replace("_train", "") + ".pt"

    # Construct the full path to the model file
    model_path = os.path.join(models_folder, model_name)

    # Check if the model file exists
    return os.path.exists(model_path), model_path
