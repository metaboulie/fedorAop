from abc import abstractmethod, ABC
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import scipy
import torch
from imblearn.combine import *
from imblearn.over_sampling import *
from imblearn.under_sampling import *
from sklearn.linear_model import LogisticRegression

from fedorAop.config import BATCH_SIZE


def generate_probabilities(
    size: int,
    distribution: scipy.stats.rv_continuous = scipy.stats.norm,
    **kwargs,
) -> np.ndarray:
    """Generate an 1D-array filled with probabilities from a given distribution

    Parameters
    ----------
    size : int
        The size of the output array.
    distribution : scipy.stats.rv_continuous, optional
        The distribution to generate each probability from, by default scipy.stats.norm
    Returns
    -------
    np.ndarray
        An 1D-array filled with probabilities from the given distribution
    """
    return scipy.special.softmax(distribution.rvs(size=size, **kwargs), axis=0)


def average_rank(df: pd.DataFrame, _columns: list[str], mode: str = "descending") -> pd.DataFrame:
    """
    Calculate the average rank for each column in the given DataFrame.

    Parameters
    ---------
    df: pd.DataFrame
        The DataFrame containing the columns to calculate average rank for.
    _columns: list[str]
        The list of column names to calculate average rank for.
    mode : str, optional
        The mode for ranking the columns. Either "ascending" or "descending". The default is "descending".
    Returns
    ------
    pd.DataFrame
        The DataFrame with an additional column "Average_rank" containing the average rank for each column.
    """
    # Create a copy of the DataFrame to avoid modifying the original DataFrame
    df_copy = df.copy()
    # If the data frame doesn't have the "Average Rank" column, add it
    if "Average Rank" not in df_copy.columns:
        df_copy["Average Rank"] = 0
    # Sum the ranks of the selected columns, divide by the number of columns, and add it to the "Average Rank" column
    if mode == "descending":
        df_copy["Average Rank"] += df_copy[_columns].rank(ascending=False).sum(axis=1) / len(_columns)
    else:
        df_copy["Average Rank"] += df_copy[_columns].rank(ascending=True).sum(axis=1) / len(_columns)
    return df_copy


def get_under_sampler(sampler: str, **kwargs):
    """
    Generate an under-sampling-sampler based on the given sampler type.

    Parameters:
        sampler (str): The type of sampler to generate.

    Returns:
        Undersampler: An under-sampling-sampler object based on the given sampler type.

    Raises:
        ValueError: If an invalid sampler type is provided.
    """
    match sampler:
        case "ClusterCentroids":
            return ClusterCentroids(**kwargs)
        case "CondensedNearestNeighbour":
            return CondensedNearestNeighbour(n_jobs=-1, **kwargs)
        case "EditedNearestNeighbours":
            return EditedNearestNeighbours(n_jobs=-1, **kwargs)
        case "RepeatedEditedNearestNeighbours":
            return RepeatedEditedNearestNeighbours(n_jobs=-1, **kwargs)
        case "AllKNN":
            return AllKNN(n_jobs=-1, **kwargs)
        case "InstanceHardnessThreshold":
            # Check if the `estimator` parameter is set
            if "estimator" not in kwargs:
                kwargs["estimator"] = LogisticRegression(max_iter=1000000, solver="lbfgs")
            return InstanceHardnessThreshold(n_jobs=-1, **kwargs)
        case "NearMiss1":
            return NearMiss(version=1, n_jobs=-1, **kwargs)
        case "NearMiss2":
            return NearMiss(version=2, n_jobs=-1, **kwargs)
        case "NearMiss3":
            return NearMiss(version=3, n_jobs=-1, **kwargs)
        case "NeighbourhoodCleaningRule":
            return NeighbourhoodCleaningRule(n_jobs=-1, **kwargs)
        case "OneSidedSelection":
            return OneSidedSelection(n_jobs=-1, **kwargs)
        case "RandomUnderSampler":
            return RandomUnderSampler(**kwargs)
        case "TomekLinks":
            return TomekLinks(n_jobs=-1, **kwargs)
        case _:
            raise ValueError(f"Invalid sampler :{sampler}")


def get_over_sampler(sampler: str, **kwargs):
    """
    Given a sampler as input, this function returns an over-sampler object based on the input value.

    Parameters:
        sampler (str): The name of the oversampling technique to use.

    Returns:
        An over-sampler object based on the input value.
        If the input value is not a valid sampler, a ValueError is raised.
    """
    match sampler:
        case "RandomOverSampler":
            return RandomOverSampler(**kwargs)
        case "SMOTE":
            return SMOTE(n_jobs=-1, **kwargs)
        case "SMOTENC":
            return SMOTENC(n_jobs=-1, **kwargs)
        case "SMOTEN":
            return SMOTEN(n_jobs=-1, **kwargs)
        case "ADASYN":
            return ADASYN(n_jobs=-1, **kwargs)
        case "BorderlineSMOTE1":
            return BorderlineSMOTE(n_jobs=-1, kind="borderline-1", **kwargs)
        case "BorderlineSMOTE2":
            return BorderlineSMOTE(n_jobs=-1, kind="borderline-2", **kwargs)
        case "KMeansSMOTE":
            return KMeansSMOTE(n_jobs=-1, **kwargs)
        case "SVMSMOTE":
            return SVMSMOTE(n_jobs=-1, **kwargs)
        case _:
            raise ValueError(f"Invalid sampler :{sampler}")


def get_combined_sampler(sampler: str, **kwargs):
    """
    Generate a combined sampler based on the input string.

    Parameters:
    sampler (str): The name of the sampler to generate. Valid options are "SMOTEENN" and "SMOTETomek".

    Returns:
    obj: An instance of the combined sampler.

    Raises:
    ValueError: If the input sampler is not a valid option.
    """
    match sampler:
        case "SMOTEENN":
            return SMOTEENN(n_jobs=-1, **kwargs)
        case "SMOTETomek":
            return SMOTETomek(n_jobs=-1, **kwargs)
        case _:
            raise ValueError(f"Invalid sampler: {sampler}")


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
    assert isinstance(data, np.ndarray), "The type of the input data must be numpy.ndarray"

    # Split the data into feature-matrix and label-series
    X = data[:, :-1]
    y = data[:, -1]

    # Convert the feature-matrix and label-series to tensors
    X_tensor = torch.tensor(X, dtype=torch.float32, requires_grad=True)
    y_tensor = torch.tensor(y, dtype=torch.long)  # The dtype of y must be `torch.long`

    return X_tensor, y_tensor


@dataclass
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
    num_labels: int
        The number of different labels in the input data
    choices: list[int]
        The indexes of the batch_data in the original data
    """

    batch_size: int = BATCH_SIZE
    data: np.ndarray = field(default_factory=np.ndarray, init=True)
    size: int = field(init=False)
    num_labels: int = field(init=False)
    choices: list = field(default_factory=list, init=False)

    def __post_init__(self):
        """Initialize the number of the observations and unique labels in the input data"""
        self.size = self.data.shape[0]
        self.num_labels = len(np.unique(self.data[:, -1]))
        pass

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
    num_labels: int
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

    distribution: scipy.stats.rv_continuous = field(default=scipy.stats.uniform)
    weights: np.ndarray[float] = field(init=False)

    def init_distribution(self, **kwargs) -> None:
        """Initialize the distribution of the weights"""
        self.distribution = scipy.stats.rv_continuous(**kwargs)

    def sample(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Use the generated weights to sample the data, overrides from the `sample` method from the superClass Sample

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            X_Batch, y_Batch
        """
        self.weights = generate_probabilities(distribution=self.distribution, size=self.size)
        self.choices = np.random.choice(range(self.size), self.batch_size, False, self.weights)

        return feature_label_split(self.data[self.choices])


@dataclass
class Bootstrap(Sample):
    """This class inherits from Sample, see the doc of Sample for more details

    This class split batch_size into num_labels groups and sample each label relevant count of observations by Bootstrap

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
    num_labels: int
        The number of different labels in the input data
    choices: list[int]
        The indexes of the batch_data in the original data
    change_indexes: list
        The list of the indexes where the label of the sorted data changes

    Usages:
    ------
    bootstrapModel = Bootstrap(data=data)
    X_batch, y_batch = bootstrapModel.sample()
    """

    change_indexes: list = field(init=False)

    def __post_init__(self):
        """Sort the input data by their labels and record the indexes where the label changes"""
        self.size = self.data.shape[0]
        self.num_labels = len(np.unique(self.data[:, -1]))
        self.data = self.data[self.data[:, -1].argsort()]
        self.change_indexes = list(np.where(np.diff(self.data[:, -1]))[0] + 1)
        self.change_indexes.append(self.data.shape[0])
        self.change_indexes.insert(0, 0)

    @property
    def get_num(self) -> np.ndarray:
        """Split the range of batch_size to num_labels groups, the count of numbers in each group accords
        how many observations should be sampled by Bootstrap for each label

        Returns
        -------
        np.ndarray
            The array of how many observations should be sampled by Bootstrap for each label
        """
        nums = sorted(np.random.choice(range(1, self.batch_size), self.num_labels - 1, False))
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
        nums = self.get_num
        for i in range(len(self.change_indexes) - 1):
            self.choices += list(
                np.random.choice(
                    range(self.change_indexes[i], self.change_indexes[i + 1]),
                    nums[i],
                    True,
                )
            )
        return feature_label_split(self.data[self.choices])


@dataclass
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
    num_labels: int
        The number of different labels in the input data
    choices: list[int]
        The indexes of the batch_data in the original data
    change_indexes: list
        The list of the indexes where the label of the sorted data changes
    max_num: int
        The count of observations of the most frequent label

    Usages:
    -------
    SWIModel = SampleWithImputation(data=data)
    SWIModel.iterLabels()
    X_Batch, y_Batch = SWIModel.sample
    """

    max_num: int = field(init=False)

    def __post_init__(self):
        """Sort the input data by their labels, record the indexes where the label changes
        and get the count of observations of the most frequent label"""

        self.size = self.data.shape[0]
        self.num_labels = len(np.unique(self.data[:, -1]))
        self.data = self.data[self.data[:, -1].argsort()]
        self.change_indexes = list(np.where(np.diff(self.data[:, -1]))[0] + 1)
        self.change_indexes.append(self.data.shape[0])
        self.change_indexes.insert(0, 0)
        self.max_num = max(np.diff(self.change_indexes))

    def iter_labels(self) -> None:
        """Iterate each label, and impute data"""
        for i in range(len(self.change_indexes) - 1):
            num_imputation = (
                self.max_num - self.change_indexes[i + 1] + self.change_indexes[i]
            )  # The number of the data to be imputed equals self.max_num minus the count of observations of this label
            label_mean, label_std = self.feature_stats_agg(label_counter=i)
            imputedData = self.impute_data(i, num_imputation, label_mean, label_std)
            self.data = np.concatenate((self.data, imputedData), axis=0)
        self.size = self.data.shape[0]  # Update the size of the data

    @staticmethod
    def impute_data(encoded_label: int, num: int, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
        """Impute data for each label according to its means and stds for each feature

        Parameters
        ----------
        encoded_label : int
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
                np.full((num, 1), encoded_label),
            ),
            axis=1,
        )

    def feature_stats_agg(self, label_counter: int) -> tuple[np.ndarray, np.ndarray]:
        """Calculate the mean and std for each feature given a set of data with same label

        Parameters
        ----------
        label_counter: int
            A mark of the current label

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
             The mean and std for each feature of this label
        """
        label_data = self.data[
            range(self.change_indexes[label_counter], self.change_indexes[label_counter + 1])
        ]  # self.data is sorted, see the __post_init__ of Bootstrap for details
        return label_data.mean(axis=0)[:-1], label_data.std(axis=0)[:-1]

    def sample(self):
        """Sample the balanced dataset

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            X_Batch, y_Batch
        """
        self.choices = np.random.choice(range(self.size), self.batch_size, True)
        return feature_label_split(self.data[self.choices])
