import warnings

warnings.simplefilter("ignore", UserWarning)
warnings.simplefilter("ignore", FutureWarning)

from config import *
from tqdm import tqdm

from fedorAop.models.neural_network import NeuralNetwork, MLP, InputLayer, EmbedLayer, WeightedCrossEntropyLoss

from fedorAop.utils.function import (
    count_unique_labels,
    does_model_exist,
    evaluate,
    calculate_cost_matrix,
    calculate_class_weights,
)
from fedorAop.utils.io import get_data_dict

from imblearn.under_sampling import RandomUnderSampler

import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau

import json


if __name__ == "__main__":
    # Import data
    data_dict = get_data_dict(DATASET_PATH)
    datasets = list(data_dict.keys())

    # Initialize dicts for storing results for each dataset
    train_metrics_results, test_metrics_results = {}, {}

    model_info = {
        "loss": "WeightedCrossEntropyLoss",
        "activation function": "ELU",
        "sampling method": "RandomUnderSampler",
        "cost sensitive": "cost matrix",
    }

    for i in tqdm(range(len(datasets))):
        if i % 2 == 0:
            # if i != 3:
            continue

        # Get the dataset
        dataset = datasets[i]
        print(f"\nDataset: {dataset}\n")
        # Get the data
        data = data_dict[dataset]
        # Get the cost matrix
        cost_matrix = calculate_cost_matrix(data[:, -1])
        # Get the number of features
        in_features = data.shape[1] - 1
        # Get the number of labels
        n_labels = count_unique_labels(data)
        # Extract the labels
        class_labels = torch.tensor(data[:, -1], dtype=torch.int)
        # Calculate the class weights
        class_weights = calculate_class_weights(class_labels)
        # Create the loss function
        # loss_fn = nn.CrossEntropyLoss()
        loss_fn = WeightedCrossEntropyLoss(weights=class_weights)
        # Check whether there exists a trained model or not
        model_exist, model_path = does_model_exist(SAVED_MODELS_PATH, dataset)
        # If there exists a trained model, load it
        if model_exist:
            print(f"Model file exists for dataset '{dataset}'.")
            # Load the model
            model = torch.load(model_path)
            # Evaluate the model metrics on the train and test data
            train_metrics_result = evaluate(
                data_dict[datasets[i]],
                model,
                loss_fn,
                mode="model",
                cost_matrix=cost_matrix,
                **model_info,
                split="train",
            )
            test_metrics_result = evaluate(
                data_dict[datasets[i - 1]],
                model,
                loss_fn,
                mode="model",
                cost_matrix=cost_matrix,
                **model_info,
                split="test",
            )

            train_metrics_results[datasets[i]] = train_metrics_result
            test_metrics_results[datasets[i - 1]] = test_metrics_result

            method_path = f"{RESULTS_PATH}/METHOD5.json"
            with open(method_path, "r") as json_file:
                if json_file.read():
                    json_file.seek(0)
                    result = json.load(json_file)
                else:
                    result = {}
            result.update(train_metrics_results)
            result.update(test_metrics_results)

            with open(method_path, "w") as json_file:
                json.dump(result, json_file)

            continue
        # If there doesn't exist a trained model, then create one.
        else:
            # Create the model
            model = NeuralNetwork()
            # Create the input layer
            input_layer = InputLayer(in_features=in_features)
            # Create the embedding layer
            embed_layer = EmbedLayer(in_features=in_features, out_features=300)
            # Create the MLP
            mlp_layers = MLP(in_features=300, out_features=n_labels, n_layers=3, softmax=False)
            # Combine the layers
            model.buildLayers(input_layer, embed_layer, mlp_layers)

        # Number of steps in each epoch
        n_steps = data.shape[0] // BATCH_SIZE
        # Initialize the optimizer to do the gradient descent
        optimizer = torch.optim.Adam(model.parameters(), ADAM_LR, betas=(ADAM_BETA1, ADAM_BETA2), eps=ADAM_EPS)
        # Initialize the Learning-rate Scheduler
        scheduler = ReduceLROnPlateau(
            optimizer, "min", patience=EARLY_STOPPING_PATIENCE, threshold=EARLY_STOPPING_THRESHOLD
        )
        # sample_model = SampleWithImputation(data=data)
        # sample_model.iter_labels()
        sample_model = RandomUnderSampler()
        X, y = data[:, :-1], data[:, -1]

        for epoch in tqdm(range(N_EPOCHS)):
            count = 0
            print(f"Epoch {epoch + 1}\n-------------------------------")
            model.train()

            for _ in tqdm(range(n_steps)):
                count += 1
                # Sample the data
                # X_train, y_train = sample_model.sample()
                X_train, y_train = sample_model.fit_resample(X, y)
                X_train, y_train = torch.tensor(X_train, dtype=torch.float32, requires_grad=True), torch.tensor(
                    y_train, dtype=torch.long
                )
                # Train the model
                optimizer.zero_grad()
                pred = model(X_train)
                loss = loss_fn(pred, y_train)
                loss.backward()
                optimizer.step()
                if count % N_STEPS_TO_PRINT == 0:
                    evaluate(data=data, model=model, loss_fn=loss_fn, mode="step", cost_matrix=cost_matrix)

            evaluate(data=data, model=model, loss_fn=loss_fn, mode="train", cost_matrix=cost_matrix)
            evaluate(
                data=data_dict[datasets[i - 1]], model=model, loss_fn=loss_fn, mode="test", cost_matrix=cost_matrix
            )
        # Save the trained model
        torch.save(model, model_path)
        print("Model is saved")
