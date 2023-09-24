from dataclasses import dataclass, field
from abc import ABCMeta, abstractmethod, ABC
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
    try:
        assert isinstance(data, np.ndarray)
    except AssertionError as e:
        e.add_note("The type of the input data must be numpy.ndarray")
        raise

    X = data[:, :-1]
    y = data[:, -1]

    X_tensor, y_tensor = torch.tensor(
        X, dtype=torch.float32, requires_grad=True
    ), torch.tensor(y, dtype=torch.long)

    return X_tensor, y_tensor


@dataclass()
class Sample(ABC):
    batch_size: int = 64
    data: np.ndarray = field(default_factory=np.ndarray, repr=False)
    size: int = field(init=False)
    choices: list = field(default_factory=list, init=False, repr=False)

    def __post_init__(self) -> None:
        self.size = self.data.shape[0]

    @abstractmethod
    def sample(self):
        pass

    @property
    def countUniqueLabels(self) -> int:
        # Extract the last column of the 2D NumPy array
        labels_column = self.data[:, -1]

        # Find the unique labels
        unique_labels = np.unique(labels_column)

        return len(unique_labels)

    @property
    def getNum(self):
        numLabels = self.countUniqueLabels
        nums = sorted(np.random.choice(range(1, self.batch_size), numLabels - 1, False))
        nums.append(self.batch_size)
        nums.insert(0, 0)
        return np.diff(nums)


@dataclass()
class Resample(Sample):
    """
    Class for generating a batch-sampling model
    Usage:
    sampleModel = Resample(data)
    X, y = sampleModel.sample(distribution, *args)
    """

    pros: np.ndarray[float] = field(init=False, repr=False)

    def sample(
        self, distribution: scipy.stats.rv_continuous = scipy.stats.uniform, *args
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        self.pros = prosGenerator(distribution=distribution, size=self.size, *args)

        self.choices = np.random.choice(
            range(self.size), self.batch_size, False, self.pros
        )

        return featureLabelSplit(self.data[self.choices])


@dataclass()
class Bootstrap(Sample):
    """
    Usage:
    sampleModel = Bootstrap(data=data)
    X, y = sampleModel.sample
    """

    changeIndexes: list = field(init=False)

    def sortDataByLabels(self):
        self.data = self.data[self.data[:, -1].argsort()]
        self.changeIndexes = list(np.where(np.diff(self.data[:, -1]))[0] + 1)
        self.changeIndexes.append(self.data.shape[0])
        self.changeIndexes.insert(0, 0)

    @property
    def sample(self):
        self.sortDataByLabels()
        nums = self.getNum
        for i in range(len(self.changeIndexes) - 1):
            self.choices += list(
                np.random.choice(
                    range(self.changeIndexes[i], self.changeIndexes[i + 1]),
                    nums[i],
                    True,
                )
            )
        return featureLabelSplit(self.data[self.choices])


@dataclass()
class DeepResample(Resample):
    """
    Usage:

    """

    labels: List[int] | np.ndarray[int] = field(default=None)
    lamb: float = 0.2

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
