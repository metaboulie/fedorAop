from dataclasses import dataclass, field
from typing import List

import numpy as np
import scipy
import torch


def prosGenerator(
    distribution: scipy.stats.rv_continuous = scipy.stats.norm, size: int = 100, *args
) -> np.array:
    """
    Generate an 1D-array filled with probabilities.
    param distribution: how to generate each probability
    param size: how many probabilities should be generated, should be equal to the size of the entire dataset
    """
    return scipy.special.softmax(distribution.rvs(size=size, *args))


def featureLabelSplit(data: np.array) -> torch.tensor:
    """Given a dataset, split it into feature-set and target-set"""
    assert isinstance(data, np.ndarray)

    X = data[:, :-1]
    y = data[:, -1]

    X_tensor, y_tensor = torch.tensor(
        X, dtype=torch.float32, requires_grad=True
    ), torch.tensor(y, dtype=torch.long)

    return X_tensor, y_tensor


@dataclass()
class Resample:
    """
    Class for generating a batch-sampling model
    Usage:
    sampleModel = Resample(data)
    X, y = sampleModel.sample(distribution, *args)
    """

    batch_size: int = 64
    data: np.array = field(default_factory=np.array, repr=False)
    size: int = field(init=False)
    pros: List[float] = field(init=False, repr=False)
    choices: np.array = field(default_factory=np.array, init=False, repr=False)

    def __post_init__(self):
        self.size = self.data.shape[0]

    def sample(
        self, distribution: scipy.stats.rv_continuous = scipy.stats.uniform, *args
    ):
        self.pros = prosGenerator(distribution=distribution, size=self.size, *args)

        self.choices = np.random.choice(
            range(self.size), self.batch_size, False, self.pros
        )

        return featureLabelSplit(self.data[self.choices])


@dataclass()
class DeepResample(Resample):
    """
    Usage:

    """

    labels: List[int] = field(default_factory=list)
    lamb: float = 0.2

    def __post_init__(self):
        self.size = self.data.shape[0]

    def sample(
        self, distribution: scipy.stats.rv_continuous = scipy.stats.uniform, *args
    ):
        self.pros = prosGenerator(distribution=distribution, size=self.size, *args)
        self.choices = np.random.choice(
            range(self.size), self.batch_size, False, self.pros
        )

    @property
    def resample(self):
        return np.random.choice(
            self.choices, np.ceil(self.lamb * self.batch_size), False
        )

    def changeLabel(self):
        self.data[:, -1][self.resample] = np.random.choice(
            self.labels, np.ceil(self.lamb * self.batch_size)
        )

    @property
    def noiseGenerator(self):
        return NotImplemented

    def changeFeature(self):
        pass

    @property
    def dataSplit(self):
        return featureLabelSplit(self.data)
