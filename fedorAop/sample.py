from abc import abstractmethod, ABC
from dataclasses import dataclass, field
import numpy as np
import scipy
import torch


def prosGenerator(
    distribution: scipy.stats.rv_continuous = scipy.stats.norm,
    size: int = 100,
    *args,
) -> np.ndarray:
    """Generate an 1D-array filled with probabilities from a given distribution

    Parameters
    ----------
    distribution : scipy.stats.rv_continuous, optional
        The distribution to generate each probability from, by default scipy.stats.norm
    size : int, optional
        How many probabilities should be generated, by default 100

    Returns
    -------
    np.ndarray
        An 1D-array filled with probabilities from the given distribution
    """
    return scipy.special.softmax(distribution.rvs(size=size, *args))


def feature_label_split(data: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Split the given dataset into feature-matrix and label-series, i.e. X and y.

    Parameters
    ----------
    data : np.ndarray
        The data to be split.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        X, y
    """
    # Ensure that the input data is of type np.ndarray
    assert isinstance(
        data, np.ndarray
    ), "The type of the input data must be numpy.ndarray"

    # Split the data into feature-matrix and label-series
    X = data[:, :-1]
    y = data[:, -1]

    # Convert the feature-matrix and label-series to tensors
    X_tensor = torch.tensor(X, dtype=torch.float32, requires_grad=True)
    y_tensor = torch.tensor(y, dtype=torch.long)  # The dtype of y must be `torch.long`

    return X_tensor, y_tensor


@dataclass()
class Sample(ABC):
    """Base class for the following classes to sample the input data for BGD

    Parameters:
    ----------
    batch_size: int, optional
        The size of the batch, by default 64
    data: np.ndarray
        The data to be sampled

    Attributes:
    ----------
    batch_size: int, optional
        The size of the batch, by default 64
    data: np.ndarray
        The data to be sampled
    size: int
        The count of observations in the data
    numLabels: int
        The number of different labels in the input data
    choices: list[int]
        The indexes of the batch_data in the original data
    """

    batch_size: int = 64
    data: np.ndarray = field(default_factory=np.ndarray, init=True)
    size: int = field(init=False)
    numLabels: int = field(init=False)
    choices: list = field(default_factory=list, init=False)

    def __post_init__(self):
        """Initialize the number of the observations and unique labels in the input data"""
        self.size = self.data.shape[0]
        self.numLabels = len(np.unique(self.data[:, -1]))

    @property
    @abstractmethod
    def sample(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample the data for BGD in different ways in different SubClasses"""
        pass


@dataclass()
class Resample(Sample):
    """This class inherits from Sample, see the doc of Sample for more details

    Utilize a given distribution to generate weights for each observation of the input data and use these
    weights to sample the data

    Parameters:
    ----------
    batch_size: int, optional
        The size of the batch, by default 64
    data: np.ndarray
        The data to be sampled
    distribution: scipy.stats.rv_continuous, optional
        The distribution to generate each probability from, by default scipy.stats.uniform

    Attributes:
    ----------
    batch_size: int, optional
        The size of the batch, by default 64
    data: np.ndarray
        The data to be sampled
    distribution: scipy.stats.rv_continuous, optional
        The distribution to generate each probability from, by default scipy.stats.uniform
    size: int
        The count of observations in the data
    numLabels: int
        The number of different labels in the input data
    choices: list[int]
        The indexes of the batch_data in the original data
    weights: np.ndarray
        The weights for each observation

    Usages:
    ------
    resampleModel = Resample(data=data)
    X_batch, y_batch = resampleModel.sample(distribution=distribution, *args)
    """

    distribution: scipy.stats.rv_continuous = field(
        init=True, default=scipy.stats.uniform
    )
    weights: np.ndarray[float] = field(init=False)

    def sample(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Use the generated weights to sample the data, overrides from the `sample` method from the superClass Sample

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            X_Batch, y_Batch
        """
        self.weights = prosGenerator(distribution=self.distribution, size=self.size)

        self.choices = np.random.choice(
            range(self.size), self.batch_size, False, self.weights
        )

        return feature_label_split(self.data[self.choices])


@dataclass()
class Bootstrap(Sample):
    """This class inherits from Sample, see the doc of Sample for more details

    This class split batch_size into numLabels groups and sample each label relevant count of observations by Bootstrap

    Parameters:
    ----------
    batch_size: int, optional
        The size of the batch, by default 64
    data: np.ndarray
        The data to be sampled

    Attributes:
    ------------
    batch_size: int, optional
        The size of the batch, by default 64
    data: np.ndarray
        The data to be sampled
    size: int
        The count of observations in the data
    numLabels: int
        The number of different labels in the input data
    choices: list[int]
        The indexes of the batch_data in the original data
    changeIndexes: list
        The list of the indexes where the label of the sorted data changes

    Usages:
    ------
    bootstrapModel = Bootstrap(data=data)
    X_batch, y_batch = bootstrapModel.sample()
    """

    changeIndexes: list = field(init=False)

    def __post_init__(self):
        """Sort the input data by their labels and record the indexes where the label changes"""
        self.size = self.data.shape[0]
        self.numLabels = len(np.unique(self.data[:, -1]))
        self.data = self.data[self.data[:, -1].argsort()]
        self.changeIndexes = list(np.where(np.diff(self.data[:, -1]))[0] + 1)
        self.changeIndexes.append(self.data.shape[0])
        self.changeIndexes.insert(0, 0)

    @property
    def getNum(self) -> np.ndarray:
        """Split the range of batch_size to numLabels groups, the count of numbers in each group accords
        how many observations should be sampled by Bootstrap for each label

        Returns
        -------
        np.ndarray
            The array of how many observations should be sampled by Bootstrap for each label
        """
        nums = sorted(
            np.random.choice(range(1, self.batch_size), self.numLabels - 1, False)
        )
        nums.append(self.batch_size)
        nums.insert(0, 0)
        return np.diff(nums)

    def sample(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Utilize the generated list to sample the data by Bootstrap,
        the number of sampled observations for each label should be equal to the relevant number in the list

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            X_Batch, y_Batch
        """
        nums = self.getNum
        for i in range(len(self.changeIndexes) - 1):
            self.choices += list(
                np.random.choice(
                    range(self.changeIndexes[i], self.changeIndexes[i + 1]),
                    nums[i],
                    True,
                )
            )
        return feature_label_split(self.data[self.choices])


@dataclass()
class SampleWithImputation(Bootstrap):
    """Impute data and insert them into the trainSet to make trainSet balanced

    Parameters:
    ----------
    batch_size: int, optional
        The size of the batch, by default 64
    data: np.ndarray
        The data to be sampled

    Attributes:
    ------------
    batch_size: int, optional
        The size of the batch, by default 64
    data: np.ndarray
        The data to be sampled
    size: int
        The count of observations in the data
    numLabels: int
        The number of different labels in the input data
    choices: list[int]
        The indexes of the batch_data in the original data
    changeIndexes: list
        The list of the indexes where the label of the sorted data changes
    maxNum: int
        The count of observations of the most frequent label

    Usages:
    -------
    SWIModel = SampleWithImputation(data=data)
    SWIModel.iterLabels()
    X_Batch, y_Batch = SWIModel.sample
    """

    maxNum: int = field(init=False)

    def __post_init__(self):
        """Sort the input data by their labels, record the indexes where the label changes
        and get the count of observations of the most frequent label"""

        self.size = self.data.shape[0]
        self.numLabels = len(np.unique(self.data[:, -1]))
        self.data = self.data[self.data[:, -1].argsort()]
        self.changeIndexes = list(np.where(np.diff(self.data[:, -1]))[0] + 1)
        self.changeIndexes.append(self.data.shape[0])
        self.changeIndexes.insert(0, 0)
        self.maxNum = max(np.diff(self.changeIndexes))

    def iterLabels(self) -> None:
        """Iterate each label, and impute data"""
        for i in range(len(self.changeIndexes) - 1):
            numOfImputation = (
                self.maxNum - self.changeIndexes[i + 1] + self.changeIndexes[i]
            )  # * The number of the data to be imputed equals self.maxNum minus the count of observations of this label
            labelMean, labelStd = self.featureStatsAgg(labelCounter=i)
            imputedData = self.imputeData(i, numOfImputation, labelMean, labelStd)
            self.data = np.concatenate((self.data, imputedData), axis=0)
        self.size = self.data.shape[0]  # Update the size of the data

    @staticmethod
    def imputeData(
        encodedLabel: int, num: int, mean: np.ndarray, std: np.ndarray
    ) -> np.ndarray:
        """Impute data for each label according to its means and stds for each feature

        Parameters
        ----------
        encodedLabel : int
            The encoded label for each label
        num : int
            How many observations should be imputed
        mean : np.ndarray
            An array storing mean values for each feature
        std : np.ndarray
            An array storing std values for each feature

        Returns
        -------
        np.ndarray
            Imputed data
        """
        return np.concatenate(
            (
                np.random.normal(loc=mean, scale=std, size=(num, len(mean))),
                np.full((num, 1), encodedLabel),
            ),
            axis=1,
        )

    def featureStatsAgg(self, labelCounter: int) -> tuple[np.ndarray, np.ndarray]:
        """Calculate the mean and std for each feature given a set of data with same label

        Parameters
        ----------
        labelCounter: int
            A mark of the current label

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
             The mean and std for each feature of this label
        """
        labelData = self.data[
            range(
                self.changeIndexes[labelCounter], self.changeIndexes[labelCounter + 1]
            )
        ]  # self.data is sorted, see the __post_init__ of Bootstrap for details
        return labelData.mean(axis=0)[:-1], labelData.std(axis=0)[:-1]

    def sample(self):
        """Sample the balanced dataset

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            X_Batch, y_Batch
        """
        self.choices = np.random.choice(range(self.size), self.batch_size, True)
        return feature_label_split(self.data[self.choices])


# NotImplemented
@dataclass()
class DeepResample(Resample):
    """NotImplemented"""

    labels: list[int] | np.ndarray[int] = field(default=None)
    lamb: float = 0.2

    def sample(
        self, distribution: scipy.stats.rv_continuous = scipy.stats.uniform, *args
    ) -> None:
        self.weights = prosGenerator(distribution=distribution, size=self.size, *args)
        self.choices = np.random.choice(
            range(self.size), self.batch_size, False, self.weights
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
    def dataSplit(self) -> tuple[torch.Tensor, torch.Tensor]:
        return feature_label_split(self.data)
