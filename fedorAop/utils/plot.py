from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

sns.set_theme(style="darkgrid", palette="Set3")


def plot_loss(loss_dict: dict[str, list[float]], split: str, filename: str) -> plt.Axes:
    """
    Generates a plot of the loss values over epochs and saves it as an image.

    Parameters:
        loss_dict (dict[str, list[float]]): A dictionary mapping the names of the loss functions to their respective loss values.
        split (str): The name of the split (e.g., 'training', 'validation') for which the loss is being plotted.
        filename (str): The name of the file to save the plot image as.

    Returns:
        plt.Axes: The axes object representing the plot.

    """
    # Plot the loss
    fig, ax = plt.subplots(figsize=(16, 6))
    sns.lineplot(data=loss_dict, legend="auto", ax=ax)

    ax.set_title(f"Loss / Epoch in {split}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend(loc="best")

    # Save the plot image
    plt.savefig(f"{Path.cwd().parent.parent}/images/{filename}.png")

    return ax


# Plot the metrics results by dataset with bar groups
def plot_metrics(
    metrics_dict: dict[str, dict[str, float]], split: str, filename: str
) -> plt.Axes:
    data = pd.DataFrame(metrics_dict).transpose()
    print(data)
    ax = data.plot(kind="bar", figsize=(16, 6), rot=0)
    ax.set_title(f"Metrics by Dataset in {split}")
    ax.set_xlabel("Dataset")
    ax.set_ylabel("Value")
    ax.legend(loc="best")
    plt.savefig(f"{Path.cwd().parent.parent}/images/{filename}.png")
    return ax
