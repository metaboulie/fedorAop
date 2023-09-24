import warnings

warnings.simplefilter("ignore", UserWarning)

from config import *
from model import NeuralNetwork, MLP, InputLayer, EmbedLayer
from function import (
    countUniqueLabels,
    doesModelExist,
    evaluateTest,
    trainLoop,
    earlyStopping,
    evaluateModel,
    createLossDataframe,
    createMetricsDataframe,
)
from torch.optim.lr_scheduler import ReduceLROnPlateau
from plot import plotMetricsResults, plotLossResults
from IO import get_data_dict
import torch
from torch import nn

data_dict = get_data_dict(PATH)

datasets = list(data_dict.keys())

train_loss_results, test_loss_results, train_metrics_results, test_metrics_results = (
    [],
    [],
    [],
    [],
)


def main():
    for i in range(len(datasets)):
        # if i % 2 == 0:
        if i != 3:
            continue

        dataset = datasets[i]
        X = data_dict[dataset]

        in_features = X.shape[1] - 1
        n_features = countUniqueLabels(X)

        model_exist, model_path = doesModelExist("Models", dataset)

        if model_exist:
            # There exists a trained model
            print(f"Model file exists for dataset '{dataset}'.")
            # Load the model
            model = torch.load(model_path)

        else:
            # There doesn't exist a trained model, then create one.
            model = NeuralNetwork()
            input_layer = InputLayer(in_features=in_features)
            embed_layer = EmbedLayer(in_features=in_features, out_features=300)
            mlp_layers = MLP(in_features=300, out_features=n_features, n_layers=4)

            # Combine the layers
            model.buildLayers(input_layer, embed_layer, mlp_layers)

        ######## print(model)

        # Number of steps in each epoch
        n_steps = X.shape[0] // BATCH_SIZE

        loss_fn = nn.CrossEntropyLoss()

        train_loss_list, test_loss_list = [], []
        early_stopping_count = 0

        # Gradient descent
        optimizer = torch.optim.Adam(
            model.parameters(), LR, betas=(BETA1, BETA2), eps=EPS
        )

        # Learning-rate Scheduler
        scheduler = ReduceLROnPlateau(
            optimizer, "min", patience=PATIENCE, threshold=THRESHOLD
        )

        # Training loop
        for epoch in range(N_EPOCHS):
            count, correct, loss = 0, [], []

            print(f"Epoch {epoch + 1}\n-------------------------------")

            model.train()

            for step in range(n_steps):
                count += 1
                trainLoop(X, model, loss_fn, optimizer, count, correct, loss)

            train_loss = evaluateTest(data_dict[datasets[i]], model, loss_fn, False)
            test_loss = evaluateTest(data_dict[datasets[i - 1]], model, loss_fn)
            train_loss_list.append(train_loss)
            test_loss_list.append(test_loss)

            if earlyStopping(
                test_loss_list,
                early_stopping_count,
                patience=PATIENCE,
                threshold=THRESHOLD,
            ):
                break

            scheduler.step(test_loss)

        torch.save(model, model_path)
        print("Model is saved")

        train_metrics_result = evaluateModel(data_dict[datasets[i]], model)
        test_metrics_result = evaluateModel(data_dict[datasets[i - 1]], model)

        train_metrics_results.append(
            [dataset.replace("_train", "")] + train_metrics_result
        )
        test_metrics_results.append(
            [dataset.replace("_train", "")] + test_metrics_result
        )

        train_loss_results.append([dataset.replace("_train", "")] + train_loss_list)
        test_loss_results.append([dataset.replace("_train", "")] + test_loss_list)

    df_train_loss = createLossDataframe(train_loss_results, N_EPOCHS)
    df_test_loss = createLossDataframe(test_loss_results, N_EPOCHS)
    df_train_metrics = createMetricsDataframe(train_metrics_results)
    df_test_metrics = createMetricsDataframe(test_metrics_results)

    plotLossResults(df_train_loss, "train")
    plotLossResults(df_test_loss, "test")
    plotMetricsResults(df_train_metrics, "train")
    plotMetricsResults(df_test_metrics, "test")


if __name__ == "__main__":
    main()
