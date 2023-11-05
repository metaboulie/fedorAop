from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

sns.set_theme(style="darkgrid", palette="Set3")

from dash import Dash, dash_table
from dash.dash_table.Format import Format, Scheme


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
def plot_metrics(metrics_dict: dict[str, dict[str, float]], split: str, filename: str) -> plt.Axes:
    data = pd.DataFrame(metrics_dict).transpose()
    print(data)
    ax = data.plot(kind="bar", figsize=(16, 6), rot=0)
    ax.set_title(f"Metrics by Dataset in {split}")
    ax.set_xlabel("Dataset")
    ax.set_ylabel("Value")
    ax.legend(loc="best")
    plt.savefig(f"{Path.cwd().parent.parent}/images/{filename}.png")
    return ax


def plot_dash_table(result):
    pass
    # Initialize the Dash app
    app = Dash(__name__)

    # A list of schemas for columns of the table
    columns = [
        {"name": ["Method", "Type"], "id": "Type", "type": "text"},
        {"name": ["Method", "Sampler"], "id": "Sampler", "type": "text"},
        {
            "name": ["Dataset", "Cancer"],
            "id": "Cancer",
            "type": "numeric",
            "format": Format(scheme=Scheme.fixed, precision=4, nully="NA"),
        },
        {
            "name": ["Dataset", "FACS_CD8"],
            "id": "FACS_CD8",
            "type": "numeric",
            "format": Format(
                scheme=Scheme.fixed,
                precision=4,
                nully="NA",
            ),
        },
        {
            "name": ["Dataset", "PBMC_Batch"],
            "id": "PBMC_Batch",
            "type": "numeric",
            "format": Format(
                scheme=Scheme.fixed,
                precision=4,
                nully="NA",
            ),
        },
        {
            "name": ["Dataset", "PBMC_COVID"],
            "id": "PBMC_COVID",
            "type": "numeric",
            "format": Format(
                scheme=Scheme.fixed,
                precision=4,
                nully="NA",
            ),
        },
        {
            "name": ["Dataset", "cSCC"],
            "id": "cSCC",
            "type": "numeric",
            "format": Format(
                scheme=Scheme.fixed,
                precision=4,
                nully="NA",
            ),
        },
        {
            "name": ["Summary", "Average Rank"],
            "id": "Average Rank",
            "type": "numeric",
            "format": Format(
                scheme=Scheme.fixed,
                precision=1,
                nully="NA",
            ),
        },
    ]

    app.layout = dash_table.DataTable(  # Customize the layout
        columns=columns,
        data=result.to_dict("records"),
        merge_duplicate_headers=True,  # Merge duplicate headers
        style_as_list_view=True,  # Use list view
        style_cell={
            "textAlign": "center",  # Align text to center
        },
        style_data_conditional=(
            [
                {
                    "if": {"row_index": "odd"},  # Even rows are in a different color
                    "backgroundColor": "rgb(248, 248, 248)",
                },
            ]
            + [
                {
                    "if": {
                        "filter_query": "{{Cancer}}={}".format(_),  # Highlight the top three cells in "Cancer" column
                        "column_id": "Cancer",
                    },
                    "backgroundColor": "lightblue",
                }
                for _ in result["Cancer"].nlargest(4)
            ]
            + [
                {
                    "if": {
                        "filter_query": "{{FACS_CD8}}={}".format(
                            _
                        ),  # Highlight the top three cells in "FACS_CD8" column
                        "column_id": "FACS_CD8",
                    },
                    "backgroundColor": "lightblue",
                }
                for _ in result["FACS_CD8"].nlargest(4)
            ]
            + [
                {
                    "if": {
                        "filter_query": "{{PBMC_Batch}}={}".format(
                            _
                        ),  # Highlight the top three cells in "PBMC_Batch" column
                        "column_id": "PBMC_Batch",
                    },
                    "backgroundColor": "lightblue",
                }
                for _ in result["PBMC_Batch"].nlargest(4)
            ]
            + [
                {
                    "if": {
                        "filter_query": "{{PBMC_COVID}}={}".format(
                            _
                        ),  # Highlight the top three cells in "PBMC_COVID" column
                        "column_id": "PBMC_COVID",
                    },
                    "backgroundColor": "lightblue",
                }
                for _ in result["PBMC_COVID"].nlargest(4)
            ]
            + [
                {
                    "if": {
                        "filter_query": "{{cSCC}}={}".format(_),  # Highlight the top three cells in "cSCC" column
                        "column_id": "cSCC",
                    },
                    "backgroundColor": "lightblue",
                }
                for _ in result["cSCC"].nlargest(4)
            ]
            + [
                {
                    "if": {
                        "filter_query": "{{Average Rank}}={}".format(
                            _
                        ),  # Highlight the top three cells in "Average_rank" column
                        "column_id": "Average Rank",
                    },
                    "backgroundColor": "lightblue",
                }
                for _ in result["Average Rank"].nsmallest(4)
            ]
            + [
                {
                    "if": {
                        "filter_query": "{{Average Rank}}={}".format(_),  # Highlight the top three sampling methods
                        "column_id": "Sampler",
                    },
                    "backgroundColor": "lightblue",
                }
                for _ in result["Average Rank"].nsmallest(4)
            ]
        ),
        style_header={
            "fontWight": "bold",
            "backgroundColor": "#F5F5F5",
        },
    )

    # Run the Dash App
    app.run()
