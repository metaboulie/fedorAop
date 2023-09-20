import plotly.graph_objs as go
import plotly.offline as pyo
import polars as pl


def plotLossResults(df: pl.DataFrame, split: str):
    # Extract the dataset names and epoch columns
    dataset_names = df["Dataset"].to_list()

    # Create a list to store traces (one trace per dataset)
    traces = []

    # Loop through each dataset and create a trace
    for i, dataset_name in enumerate(dataset_names):
        x = list(range(1, df.shape[1]))
        y = df.to_numpy()[i, 1:]
        trace = go.Scatter(
            x=x,
            y=y,
            mode="lines",
            name=f"{dataset_name}",
        )
        traces.append(trace)

    # Create the layout
    layout = go.Layout(
        title=f"Loss vs Epoch_{split}",
        xaxis=dict(title="Epoch"),
        yaxis=dict(title="Loss"),
        legend=dict(orientation="h"),
    )

    # Create the figure
    fig = go.Figure(data=traces, layout=layout)

    # Show the plot
    pyo.plot(fig, filename=f"loss_vs_epoch_{split}.html")


def plotMetricsResults(df: pl.DataFrame, split: str):
    # Extract the dataset names and epoch columns
    dataset_names = df["Dataset"].to_list()
    relevant_columns = ["f1 micro", "f1 macro", "precision weighted", "recall weighted"]

    # Create a list to store traces (one trace per dataset)
    traces = []

    for dataset_name in dataset_names:
        trace = go.Bar(
            x=relevant_columns,
            y=df.filter(df["Dataset"] == dataset_name)
            .select(relevant_columns)
            .to_pandas()
            .iloc[0],
            name=dataset_name,
        )
        traces.append(trace)

    # Create the layout for the combined bar charts
    layout = go.Layout(
        barmode="group",  # Use 'group' to group bars by metric
        title=f"Metrics by Dataset_{split}",
        xaxis=dict(title="Metric"),
        yaxis=dict(title="Value"),
    )

    # Create the figure for the combined bar charts
    fig = go.Figure(data=traces, layout=layout)

    # Show the plot
    pyo.plot(fig, filename=f"metrics_by_dataset_{split}.html")
