from dataclasses import dataclass, field

import plotly.graph_objs as go
import plotly.offline as pyo
import polars as pl
from plotly.subplots import make_subplots


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


@dataclass
class DataSetStats:
    dataset: pl.DataFrame
    labelColumn: pl.Series = field(init=False)
    labelColumnName: str = field(init=False)
    labelCounts: pl.DataFrame = field(init=False)

    def __post_init__(self):
        self.labelCounts = self.countLabel

    @property
    def countLabel(self):
        self.labelColumn = self.dataset.get_column(self.dataset.columns[-1])

        self.labelColumnName = self.labelColumn.name

        try:
            assert self.labelColumn.dtype.__name__ == "Categorical"

        except AssertionError as e:
            e.add_note(
                f"The dtype of the last column of the dataset is {self.labelColumn.dtype}, "
                f"it must be polars.Categorical."
            )
            raise

        return self.labelColumn.value_counts(parallel=True).sort(self.labelColumnName)

    @classmethod
    def iterDatasets(cls, dataset_dict: dict[str, pl.DataFrame]):
        dict_iter = iter(dataset_dict.items())
        while True:
            try:
                _, df_test = next(dict_iter)
                train_name, df_train = next(dict_iter)
                dataset_name = train_name.split(sep="_t")[0]
                train_stats = cls(dataset=df_train)
                test_stats = cls(dataset=df_test)
                doublePieCharts(
                    train_stats.labelCounts[train_stats.labelColumnName],
                    train_stats.labelCounts["counts"],
                    test_stats.labelCounts["counts"],
                    dataset_name,
                )

            except StopIteration:
                break


def doublePieCharts(labels, train_values, test_values, dataset_name):
    fig = make_subplots(
        rows=1, cols=2, specs=[[{"type": "domain"}, {"type": "domain"}]]
    )
    fig.add_trace(
        go.Pie(labels=labels, values=train_values, name=f"{dataset_name} Train"), 1, 1
    )
    fig.add_trace(
        go.Pie(labels=labels, values=test_values, name=f"{dataset_name} Test"), 1, 2
    )

    # Use `hole` to create a donut-like pie chart
    fig.update_traces(
        hole=0.4, hoverinfo="label+percent+name", textinfo="label+percent"
    )

    fig.update_layout(
        title_text=f"Proportion of labels in {dataset_name} dataset",
        # Add annotations in the center of the donut pies.
        annotations=[
            dict(text="Train", x=0.18, y=0.5, font_size=20, showarrow=False),
            dict(text="Test", x=0.82, y=0.5, font_size=20, showarrow=False),
        ],
    )

    pyo.plot(fig, filename=f"Proportion of labels in {dataset_name} dataset.html")


@dataclass
class Plots:
    pass
