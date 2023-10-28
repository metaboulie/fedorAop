import warnings

warnings.simplefilter("ignore", UserWarning)

from config import *
from model import NeuralNetwork, MLP, InputLayer, EmbedLayer
from fedorAop.utils.function import (
    count_unique_labels,
    does_model_exist,
    evaluate,
    train_loop,
    early_stopping,
)
from torch.optim.lr_scheduler import ReduceLROnPlateau
from fedorAop.utils.plot import plot_loss, plot_metrics
from fedorAop.utils.io import get_data_dict
import torch
from torch import nn
from fedorAop.models.sample_models import Resample, Bootstrap, SampleWithImputation

# Import data
data_dict = get_data_dict(PATH)
datasets = list(data_dict.keys())

# Initialize dicts for storing results for each dataset
train_loss_results, test_loss_results, train_metrics_results, test_metrics_results = (
    {},
    {},
    {},
    {},
)


def main(sampleModel: str = "SampleWithImputation"):
    """
    The main function that performs the training and evaluation of a neural network model.

    Args:
        sampleModel (str): The type of sampling model to use. Default is "SampleWithImputation".
    """
    # Iterate over the datasets
    for i in range(len(datasets)):
        if i % 2 == 0:
            # if i != 3:
            continue

        # Get the dataset
        dataset = datasets[i]

        # Get the data
        data = data_dict[dataset]

        # Get the number of features
        in_features = data.shape[1] - 1

        # Get the number of labels
        n_labels = count_unique_labels(data)

        # Check whether there exists a trained model or not
        model_exist, model_path = does_model_exist("../Models", dataset)

        # If there exists a trained model, load it
        if model_exist:
            print(f"Model file exists for dataset '{dataset}'.")
            # Load the model
            model = torch.load(model_path)

        # If there doesn't exist a trained model, then create one.
        else:
            # Create the model
            model = NeuralNetwork()
            # Create the input layer
            input_layer = InputLayer(in_features=in_features)
            # Create the embedding layer
            embed_layer = EmbedLayer(in_features=in_features, out_features=300)
            # Create the MLP
            mlp_layers = MLP(in_features=300, out_features=n_labels, n_layers=3)
            # Combine the layers
            model.buildLayers(input_layer, embed_layer, mlp_layers)

        # Number of steps in each epoch
        n_steps = data.shape[0] // BATCH_SIZE

        # Loss function
        loss_fn = nn.CrossEntropyLoss()

        # Initialize two lists to record the loss
        train_loss_list, test_loss_list = [], []

        # Initialize the optimizer to do the gradient descent
        optimizer = torch.optim.Adam(model.parameters(), LR, betas=(BETA1, BETA2), eps=EPS)

        # Initialize the Learning-rate Scheduler
        scheduler = ReduceLROnPlateau(optimizer, "min", patience=PATIENCE, threshold=THRESHOLD)

        # Create the corresponding sampling model based on the sampleModel argument
        # See sample_models.py for details
        match sampleModel:
            case "Resample":
                sample_model = Resample(data=data)
            case "Bootstrap":
                sample_model = Bootstrap(data=data)
            case "SampleWithImputation":
                sample_model = SampleWithImputation(data=data)
                sample_model.iterLabels()
            case _:
                raise ValueError(f"sampleModel must be one of ['Resample', 'Bootstrap', 'SampleWithImputation']")

        # Train the model
        for epoch in range(N_EPOCHS):
            # Initialize the count the number of steps
            count = 0
            print(f"Epoch {epoch + 1}\n-------------------------------")
            model.train()

            # Train the model for N_STEPS in each epoch
            for step in range(n_steps):
                count += 1
                train_loop(data, model, loss_fn, optimizer, count, sample_model)

            # Record the loss for the current epoch for train and test data
            train_loss = evaluate(data, model, loss_fn, "train")
            test_loss = evaluate(data_dict[datasets[i - 1]], model, loss_fn, "test")
            # Append the losses to the corresponding lists
            train_loss_list.append(train_loss)
            test_loss_list.append(test_loss)

            # Check for early stopping
            if early_stopping(test_loss_list, PATIENCE, THRESHOLD):
                break

            # Update the learning rate
            scheduler.step(test_loss)

        # Save the trained model
        torch.save(model, model_path)
        print("Model is saved")

        # Store the loss_list in the corresponding dictionaries for each dataset
        train_loss_results[dataset.replace("_train", "")] = train_loss_list
        test_loss_results[dataset.replace("_train", "")] = test_loss_list

        # Evaluate the model metrics on the train and test data
        train_metrics_result = evaluate(data_dict[datasets[i]], model, loss_fn, mode="model")
        test_metrics_result = evaluate(data_dict[datasets[i - 1]], model, loss_fn, mode="model")

        # Store the metrics in the corresponding dictionaries for each dataset
        train_metrics_results[dataset.replace("_train", "")] = train_metrics_result
        test_metrics_results[dataset.replace("_train", "")] = test_metrics_result

    plot_loss(train_loss_results, "train", "train_loss")
    plot_loss(test_loss_results, "test", "test_loss")
    plot_metrics(train_metrics_results, "train", "train_metrics")
    plot_metrics(test_metrics_results, "test", "test_metrics")


if __name__ == "__main__":
    main()
