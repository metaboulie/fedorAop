from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np
import scipy
import torch


def prosGenerator(
    distribution: scipy.stats.rv_continuous = scipy.stats.norm, size: int = 100, *args
) -> np.ndarray:
    """
    Generate an 1D-array filled with probabilities.
    param distribution: how to generate each probability
    param size: how many probabilities should be generated, should be equal to the size of the entire dataset
    """
    return scipy.special.softmax(distribution.rvs(size=size, *args))


def featureLabelSplit(data: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
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
    data: np.ndarray = field(default_factory=np.ndarray, repr=False)
    size: int = field(init=False)
    pros: np.ndarray[float] = field(init=False, repr=False)
    choices: np.ndarray = field(default_factory=np.ndarray, init=False, repr=False)

    def __post_init__(self) -> None:
        self.size = self.data.shape[0]

    def sample(
        self, distribution: scipy.stats.rv_continuous = scipy.stats.uniform, *args
    ) -> Tuple[torch.Tensor, torch.Tensor]:
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

    labels: List[int] | np.ndarray[int] = field(default=None)
    lamb: float = 0.2

    def __post_init__(self) -> None:
        self.size = self.data.shape[0]

    def sample(
        self, distribution: scipy.stats.rv_continuous = scipy.stats.uniform, *args
    ) -> None:
        self.pros = prosGenerator(distribution=distribution, size=self.size, *args)
        self.choices = np.random.choice(
            range(self.size), self.batch_size, False, self.pros
        )

    @property
    def resample(self) -> np.ndarray:
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
    def dataSplit(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return featureLabelSplit(self.data)
