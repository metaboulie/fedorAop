import unittest
import pytest
import os
import plotly.graph_objects as go
from pathlib import Path

from fedorAop.plot import plot_loss, plot_metrics


import matplotlib.pyplot as plt
from pathlib import Path


def test_plot_loss():
    loss_dict = {
        "Epoch 1": [0.5, 0.4, 0.3, 0.2, 0.1],
        "Epoch 2": [0.3, 0.2, 0.1, 0.05],
        "Epoch 3": [0.2, 0.1, 0.05],
    }
    split = "Training"
    filename = "loss"

    ax = plot_loss(loss_dict, split, filename)

    # Assert that the plot is created correctly
    assert os.path.exists(Path.cwd().parent.parent / "images" / "loss.png")

    # Assert that the plot has the correct title
    assert ax.get_title() == "Loss / Epoch in Training"

    # Assert that the plot has the correct x-axis title
    assert ax.get_xlabel() == "Epoch"

    # Assert that the plot has the correct y-axis title
    assert ax.get_ylabel() == "Loss"


class TestPlotMetrics(unittest.TestCase):
    def test_plot_metrics(self):
        # Test case 2: Non-empty metrics dictionary
        metrics_dict = {
            "dataset1": {"metric1": 0.5, "metric2": 0.8},
            "dataset2": {"metric1": 0.7, "metric2": 0.9},
        }
        split = "test"
        filename = "test2"
        ax = plot_metrics(metrics_dict, split, filename)
        self.assertIsInstance(ax, plt.Axes)

        # Test case 3: Metrics dictionary with different datasets and metrics
        metrics_dict = {
            "dataset1": {"metric1": 0.5, "metric2": 0.8},
            "dataset2": {"metric1": 0.7, "metric2": 0.9},
            "dataset3": {"metric1": 0.6, "metric2": 0.75},
        }
        split = "val"
        filename = "test3"
        ax = plot_metrics(metrics_dict, split, filename)
        self.assertIsInstance(ax, plt.Axes)

        # Test case 4: Metrics dictionary with duplicate datasets
        metrics_dict = {
            "dataset1": {"metric1": 0.5, "metric2": 0.8},
            "dataset2": {"metric1": 0.7, "metric2": 0.9},
        }
        split = "test"
        filename = "test4"
        ax = plot_metrics(metrics_dict, split, filename)
        self.assertIsInstance(ax, plt.Axes)


if __name__ == "__main__":
    unittest.main()
