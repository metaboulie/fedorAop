from dataclasses import dataclass, field
from typing import Any

import plotly.graph_objs as go
import plotly.offline as pyo
import polars as pl
from plotly.subplots import make_subplots


def plotLossResults(df: pl.DataFrame, split: str) -> None:
    """Plot the trace of CrossEntropy loss against epoch

    Parameters
    ----------
    df : pl.DataFrame
        Indexes accord the number of epoch and value being the loss in the relevant epoch
    split : str
        Indicate whether the loss is calculated in the train or test set
    """
    # TODO: Store the result to the images folder as *.png
    try:
        assert split == "train" or "test"
    except AssertionError as e:
        e.add_note("The value of `split` must be 'train' or 'test'")
        raise

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

    pyo.plot(fig, filename=f"loss_vs_epoch_{split}.html")


def plotMetricsResults(
        df: pl.DataFrame, split: str, metricNames: list[str] = None
) -> None:
    """Plot the evaluated metrics, each bar group is for a specific metric and
    in each group several datasets are aggregated
    Parameters
    ----------
    df : pl.DataFrame
        Schema is ['Dataset_name', **metric_names], each row is the name of a dataset following the values
        of each metric
    split : str
        Indicate whether the metrics are calculated in the train or test set
    metricNames : list[str], optional
        The names of the evaluated metrics, by default None
    """
    # TODO: Store the result to the images folder as *.png
    # Extract the dataset names and epoch columns
    dataset_names = df["Dataset"].to_list()
    if metricNames is None:
        metricNames = df.columns[1:]
    else:
        try:
            assert metricNames == df.columns[1:]
        except AssertionError as e:
            e.add_note("Make sure the input metricNames is in the correct sequence")
            print(e)

    # Create a list to store traces (one trace per dataset)
    traces = []

    for dataset_name in dataset_names:
        trace = go.Bar(
            x=metricNames,
            y=df.filter(df["Dataset"] == dataset_name)
            .select(metricNames)
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

    pyo.plot(fig, filename=f"metrics_by_dataset_{split}.html")


@dataclass()
class DataSetStats:
    """Calculate stats for the input data
    Attributes:
    ------------
    dataset: pl.DataFrame
        _des
    labelColumn: pl.Series
        -des
    labelColumnName: str
        _
    labelCounts: pl.DataFrame
        _
    meanOfFeatures: dict[str, Any]
        _
    maxOfFeatures: dict[str, Any]
        _
    minOfFeatures: dict[str, Any]
        _
    nameOfLabels: list
        _
    numOfFeatures: int
        _
    """

    dataset: pl.DataFrame
    labelColumn: pl.Series = field(init=False)
    labelColumnName: str = field(init=False)
    labelCounts: pl.DataFrame = field(init=False)
    meanOfFeatures: dict[str, Any] = field(init=False, default_factory=dict)
    maxOfFeatures: dict[str, Any] = field(init=False, default_factory=dict)
    minOfFeatures: dict[str, Any] = field(init=False, default_factory=dict)
    nameOfLabels: list = field(init=False, default_factory=list)
    numOfFeatures: int = field(init=False, default_factory=int)

    def __post_init__(self):
        """Calculate the number of features in the input data, get the number of observations for each label, and"""
        self.numOfFeatures = self.dataset.shape[1] - 1
        self.labelCounts = self.countLabel
        self.aggTargets()

    @property
    def countLabel(self) -> pl.DataFrame:
        """Extract the label column and the name of the label column in the input data, count the number of observations
        for each label
        Returns
        -------
        pl.DataFrame
            The number of observations for each label
        """
        self.labelColumn = self.dataset.get_column(self.dataset.columns[-1])

        self.labelColumnName = self.labelColumn.name

        try:
            assert self.labelColumn.dtype.__name__ == "Categorical"

        except AssertionError as e:
            e.add_note(
                f"The dtype of the last column of the dataset is {self.labelColumn.dtype}, "
                f"it should be polars.Categorical."
            )
            print(e)

        return self.labelColumn.value_counts(parallel=True).sort(self.labelColumnName)

    def aggTargets(self):
        for targetGroup in self.dataset.partition_by(self.labelColumnName):
            self.nameOfLabels.append(targetGroup[self.labelColumnName][0])
            self.meanOfFeatures[
                targetGroup[self.labelColumnName][0]
            ] = targetGroup.mean().row(0)[:-1]
            self.maxOfFeatures[
                targetGroup[self.labelColumnName][0]
            ] = targetGroup.max().row(0)[:-1]
            self.minOfFeatures[
                targetGroup[self.labelColumnName][0]
            ] = targetGroup.min().row(0)[:-1]

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
                aggPlot(
                    train_stats.nameOfLabels,
                    list(range(1, train_stats.numOfFeatures + 1)),
                    train_stats.meanOfFeatures,
                    train_stats.maxOfFeatures,
                    train_stats.minOfFeatures,
                    test_stats.meanOfFeatures,
                    test_stats.maxOfFeatures,
                    test_stats.minOfFeatures,
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

    # fig.write_image(f"images/Proportion of labels in {dataset_name} dataset.png")
    pyo.plot(fig, filename=f"Proportion of labels in {dataset_name} dataset.html")


def aggPlot(
        nameOfLabels,
        x: list,
        y1: dict[str, tuple],
        y1_upper: dict[str, tuple],
        y1_lower: dict[str, tuple],
        y2: dict[str, tuple],
        y2_upper: dict[str, tuple],
        y2_lower: dict[str, tuple],
        dataset_name: str,
):
    x_rev = x[::-1]

    fig = make_subplots(
        rows=len(nameOfLabels),
        cols=1,
        shared_xaxes=True,
        subplot_titles=nameOfLabels,
    )

    for i, nameOfLabel in enumerate(nameOfLabels):
        # fig.add_trace(
        #     go.Scatter(
        #         x=x+x_rev,
        #         y=y1_upper[nameOfLabel] + y1_lower[nameOfLabel],
        #         fill="toself",
        #         fillcolor="rgba(0,100,80,0.2)",
        #         line_color="rgba(255,255,255,0)",
        #         showlegend=False,
        #         name="Train",
        #     ),
        #     row=i+1,
        #     col=1,
        # )
        # fig.add_trace(
        #     go.Scatter(
        #         x=x + x_rev,
        #         y=y2_upper[nameOfLabel] + y2_lower[nameOfLabel],
        #         fill="toself",
        #         fillcolor="rgba(0,176,246,0.2)",
        #         line_color="rgba(255,255,255,0)",
        #         showlegend=False,
        #         name="Test",
        #     ),
        #     row=i + 1,
        #     col=1,
        # )
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y1[nameOfLabel],
                line_color="rgba(0,100,80,0.6)",
                name="Train",
            ),
            row=i + 1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y2[nameOfLabel],
                line_color="rgba(0,176,246,0.6)",
                name="Test",
            ),
            row=i + 1,
            col=1,
        )

    fig.update_layout(
        height=1000 * len(nameOfLabels),
        showlegend=False,
        title_text=f"Comparison of different cell types in {dataset_name} dataset",
    )

    pyo.plot(
        fig,
        filename=f"Comparison of different cell types in {dataset_name} dataset.html",
    )


@dataclass
class Plots:
    pass
