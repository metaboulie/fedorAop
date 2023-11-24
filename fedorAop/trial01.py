import warnings

warnings.simplefilter("ignore", UserWarning)

from imblearn.combine import SMOTEENN
import torch
import torch.nn as nn
from tqdm import tqdm

from config import *
from fedorAop.models.neural_network import (
    NeuralNetwork,
    MLP,
    DRLayers,
    InputLayer,
    WeightedCrossEntropyLoss,
    EmbedLayer,
    FocalLoss,
)
from fedorAop.models.sampling import SampleWithImputation, Bootstrap, Resample
from fedorAop.utils.function import (
    count_unique_labels,
    does_model_exist,
    evaluate,
    calculate_cost_matrix,
    # early_stopping,
)

from torch.optim.lr_scheduler import ReduceLROnPlateau

from fedorAop.utils.plot import plot_metrics
from fedorAop.utils.io import get_data_dict


def train_phase_one(_data, _model, _loss_fn, _optimizer, _n_steps, _cost_matrix):
    # Sample the data
    # sample_model = RandomOverSampler()

    sample_model = SampleWithImputation(data=_data)
    sample_model.iter_labels()
    # X, y = data[:, :-1], data[:, -1]

    for epoch in tqdm(range(N_EPOCHS)):
        count = 0
        print(f"Epoch {epoch + 1}\n-------------------------------")
        _model.train()

        for _ in tqdm(range(_n_steps)):
            count += 1
            # sample_model.sample()
            X_train, y_train = sample_model.sample()
            # X_train, y_train = sample_model.fit_resample(X, y)
            # X_train = torch.tensor(X_train, dtype=torch.float32, requires_grad=True)
            # y_train = torch.tensor(y_train, dtype=torch.long)
            # Train the model
            optimizer.zero_grad()
            pred = _model(X_train)
            loss = _loss_fn(pred, y_train)
            loss.backward()
            optimizer.step()
            if count % N_STEPS_TO_PRINT == 0:
                evaluate(data=_data, model=_model, loss_fn=_loss_fn, mode="step", cost_matrix=_cost_matrix)

        evaluate(data=_data, model=_model, loss_fn=_loss_fn, mode="train", cost_matrix=_cost_matrix)
        evaluate(data=data_dict[datasets[i - 1]], model=_model, loss_fn=_loss_fn, mode="test", cost_matrix=_cost_matrix)


def train_phase_two(_data, _model, _loss_fn, _optimizer, _n_steps, _cost_matrix):
    X, y = _data[:, :-1], _data[:, -1]
    _model.layers[0].frozen()
    _mlp = _model.layers[1]
    X = torch.tensor(X, dtype=torch.float32, requires_grad=False)
    y = torch.tensor(y, dtype=torch.long)
    X = _model.layers[0].forward(X)
    sample_model = SMOTEENN(n_jobs=-1)
    # sample_model = RandomUnderSampler()
    for epoch in tqdm(range(N_EPOCHS)):
        count = 0
        print(f"Epoch {epoch + 1}\n-------------------------------")
        _mlp.train()

        for _ in tqdm(range(_n_steps)):
            count += 1
            X_train, y_train = sample_model.fit_resample(X, y)
            X_train = torch.tensor(X_train, dtype=torch.float32, requires_grad=True)
            y_train = torch.tensor(y_train, dtype=torch.long)
            # Train the model
            optimizer.zero_grad()
            pred = _mlp(X_train)
            loss = _loss_fn(pred, y_train)
            loss.backward()
            optimizer.step()
            if count % N_STEPS_TO_PRINT == 0:
                _model.layers[1] = _mlp
                evaluate(data=_data, model=_model, loss_fn=_loss_fn, mode="step", cost_matrix=_cost_matrix)

        _model.layers[1] = _mlp

        evaluate(data=_data, model=_model, loss_fn=_loss_fn, mode="train", cost_matrix=_cost_matrix)
        evaluate(data=data_dict[datasets[i - 1]], model=_model, loss_fn=_loss_fn, mode="test", cost_matrix=_cost_matrix)
    pass


if __name__ == "__main__":
    # Import data
    data_dict = get_data_dict(DATASET_PATH)
    print(f"Number of datasets: {len(data_dict)}\n")
    datasets = list(data_dict.keys())

    # Initialize dicts for storing results for each dataset
    train_metrics_results, test_metrics_results = (
        {},
        {},
    )
    for i in range(len(datasets)):
        # if i % 2 == 0:  # Skip the test sets
        if i != 3:
            continue

        # Get the dataset
        dataset = datasets[i]
        print(f"Dataset: {dataset}\n")

        # Get the data
        data = data_dict[dataset]

        cost_matrix = calculate_cost_matrix(data[:, -1])
        print(f"Cost Matrix: {cost_matrix}\n\n")

        # Get the number of features
        in_features = data.shape[1] - 1

        # Get the number of labels
        n_labels = count_unique_labels(data)

        # Check whether there exists a trained model or not
        model_exist, model_path = does_model_exist("../saved_models", dataset)

        # If there exists a trained model, load it
        if model_exist:
            print(f"Model file exists for dataset '{dataset}'.\n")
            # Load the model
            model = torch.load(model_path)

        # If there doesn't exist a trained model, then create one.
        else:
            # Create the model
            model = NeuralNetwork()
            # # Create the input layer
            # input_layer = InputLayer(in_features=in_features)
            # dr_mlp_layers = MLP(in_features=input_layer.in_features, out_features=10, n_layers=3)
            # dr_layers = DRLayers(input_layer=input_layer, mlp_layers=dr_mlp_layers)
            # # Create the MLP
            # mlp_layers = MLP(in_features=dr_layers.out_features, out_features=n_labels, n_layers=3)
            # # Combine the layers
            # model.buildLayers(dr_layers, mlp_layers)
            input_layer = InputLayer(in_features=in_features)
            embed_layer = EmbedLayer(in_features=in_features, out_features=300)
            mlp_layers = MLP(in_features=300, out_features=n_labels, n_layers=3, softmax=False)
            model.buildLayers(input_layer, embed_layer, mlp_layers)

        # Number of steps in each epoch
        n_steps = data.shape[0] // BATCH_SIZE
        class_labels = torch.tensor(data[:, -1], dtype=torch.int)
        class_counts = torch.bincount(class_labels)
        class_weights = max(class_counts) / class_counts
        print(f"Class Weights: {class_weights}\n\n")
        # Loss function
        # loss_fn = WeightedCrossEntropyLoss(weights=class_weights)
        loss_fn = nn.CrossEntropyLoss()
        # loss_fn = FocalLoss()
        # Initialize the optimizer to do the gradient descent

        optimizer = torch.optim.Adam(model.parameters(), ADAM_LR, betas=(ADAM_BETA1, ADAM_BETA2), eps=ADAM_EPS)

        # # Initialize the Learning-rate Scheduler
        scheduler = ReduceLROnPlateau(
            optimizer, "min", patience=EARLY_STOPPING_PATIENCE, threshold=EARLY_STOPPING_THRESHOLD
        )

        sample_model = SampleWithImputation(data=data)
        sample_model.iter_labels()
        # sample_model = Bootstrap(data=data)
        # sample_model = Resample(data=data)
        # X, y = data[:, :-1], data[:, -1]

        for epoch in tqdm(range(N_EPOCHS)):
            count = 0
            print(f"Epoch {epoch + 1}\n-------------------------------")
            model.train()

            for _ in tqdm(range(n_steps)):
                count += 1
                # sample_model.sample()
                X_train, y_train = sample_model.sample()
                # X_train, y_train = sample_model.fit_resample(X, y)
                # X_train = torch.tensor(X_train, dtype=torch.float32, requires_grad=True)
                # y_train = torch.tensor(y_train, dtype=torch.long)
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

        # Train the model
        # train_phase_one(data, model, loss_fn, optimizer, n_steps, cost_matrix)
        # train_phase_two(data, model, loss_fn, optimizer, n_steps, cost_matrix)

        # Save the trained model
        # torch.save(model, model_path)
        # print("Model is saved")

        # Evaluate the model metrics on the train and test data
        train_metrics_result = evaluate(data_dict[datasets[i]], model, loss_fn, mode="model", cost_matrix=cost_matrix)
        test_metrics_result = evaluate(
            data_dict[datasets[i - 1]], model, loss_fn, mode="model", cost_matrix=cost_matrix
        )

        # Store the metrics in the corresponding dictionaries for each dataset
        train_metrics_results[dataset.replace("_train", "")] = train_metrics_result
        test_metrics_results[dataset.replace("_train", "")] = test_metrics_result

    plot_metrics(train_metrics_results, "train", "train_metrics")
    plot_metrics(test_metrics_results, "test", "test_metrics")
