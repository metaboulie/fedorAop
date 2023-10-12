import warnings

warnings.simplefilter("ignore", UserWarning)
warnings.simplefilter("ignore", FutureWarning)

import logging
from dash import Dash, dash_table
from dash.dash_table.Format import Format, Scheme
import pandas as pd
from tqdm import tqdm
import numpy as np


from imblearn.under_sampling import (
    ClusterCentroids,
    CondensedNearestNeighbour,
    EditedNearestNeighbours,
    RepeatedEditedNearestNeighbours,
    AllKNN,
    InstanceHardnessThreshold,
    NearMiss,
    NeighbourhoodCleaningRule,
    OneSidedSelection,
    RandomUnderSampler,
    TomekLinks,
)

from imblearn.over_sampling import (
    RandomOverSampler,
    SMOTE,
    SMOTENC,
    SMOTEN,
    ADASYN,
    BorderlineSMOTE,
    KMeansSMOTE,
    SVMSMOTE,
)

from imblearn.combine import SMOTEENN, SMOTETomek
from imblearn.metrics import geometric_mean_score

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA

from fedorAop.io import get_data_dict
from fedorAop.config import PATH

# Get the datasets
data_dict = get_data_dict(PATH)
# Get the datasets' names
dataset_names = list(data_dict.keys())
# Get the indexes of the datasets
dataset_indexes = list(range(len(dataset_names)))

# Initialize dictionaries to store the data after preprocessing
X_train_dict, X_test_dict, y_train_dict, y_test_dict = {}, {}, {}, {}

# A list of under-sampling samplers
under_samplers = [
    "ClusterCentroids",
    "CondensedNearestNeighbour",
    "EditedNearestNeighbours",
    "RepeatedEditedNearestNeighbours",
    "AllKNN",
    "InstanceHardnessThreshold",
    "NearMiss1",
    "NearMiss2",
    "NearMiss3",
    "NeighbourhoodCleaningRule",
    "OneSidedSelection",
    "RandomUnderSampler",
    "TomekLinks",
]

# A list of over-sampling samplers
over_samplers = [
    "RandomOverSampler",
    "SMOTE",
    "ADASYN",
    "BorderlineSMOTE1",
    "BorderlineSMOTE2",
    "KMeansSMOTE",
    "SVMSMOTE",
]

# A list of combined-sampling samplers
combined_samplers = [
    "SMOTEENN",
    "SMOTETomek",
]


def average_rank(_df: pd.DataFrame, _columns: list[str]):
    """
    Calculate the average rank for each column in the given DataFrame.

    Parameters
    ---------
    _df: pd.DataFrame
        The DataFrame containing the columns to calculate average rank for.
    _columns: list[str]
        The list of column names to calculate average rank for.

    Returns
    ------
    pd.DataFrame
        The DataFrame with an additional column "Average_rank" containing the average rank for each column.
    """
    # Calculate the average rank for each column
    for column in _columns:
        _df["Average Rank"] += _df[column].rank(ascending=False) / len(_columns)
    return _df


def get_under_sampler(sampler: str):
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
            return ClusterCentroids()
        case "CondensedNearestNeighbour":
            return CondensedNearestNeighbour(n_jobs=-1)
        case "EditedNearestNeighbours":
            return EditedNearestNeighbours(n_jobs=-1)
        case "RepeatedEditedNearestNeighbours":
            return RepeatedEditedNearestNeighbours(n_jobs=-1)
        case "AllKNN":
            return AllKNN(n_jobs=-1)
        case "InstanceHardnessThreshold":
            estimator = LogisticRegression(max_iter=1000000, solver="lbfgs")
            params = {
                "n_jobs": -1,
                "estimator": estimator,
            }
            return InstanceHardnessThreshold(**params)
        case "NearMiss1":
            return NearMiss(version=1, n_jobs=-1)
        case "NearMiss2":
            return NearMiss(version=2, n_jobs=-1)
        case "NearMiss3":
            return NearMiss(version=3, n_jobs=-1)
        case "NeighbourhoodCleaningRule":
            return NeighbourhoodCleaningRule(n_jobs=-1)
        case "OneSidedSelection":
            return OneSidedSelection(n_jobs=-1)
        case "RandomUnderSampler":
            return RandomUnderSampler()
        case "TomekLinks":
            return TomekLinks(n_jobs=-1)
        case _:
            raise ValueError("Invalid sampler")


def get_over_sampler(sampler: str):
    """
    Given a sampler as input, this function returns an over-sampler object based on the input value.

    Parameters:
        sampler (str): The name of the oversampling technique to use.

    Returns:
        An over-sampler object based on the input value. If the input value is not a valid sampler, a ValueError is raised.
    """
    match sampler:
        case "RandomOverSampler":
            return RandomOverSampler()
        case "SMOTE":
            return SMOTE(n_jobs=-1)
        case "SMOTENC":
            return SMOTENC(n_jobs=-1)
        case "SMOTEN":
            return SMOTEN(n_jobs=-1)
        case "ADASYN":
            return ADASYN(n_jobs=-1)
        case "BorderlineSMOTE1":
            return BorderlineSMOTE(n_jobs=-1, kind="borderline-1")
        case "BorderlineSMOTE2":
            return BorderlineSMOTE(n_jobs=-1, kind="borderline-2")
        case "KMeansSMOTE":
            return KMeansSMOTE(n_jobs=-1)
        case "SVMSMOTE":
            return SVMSMOTE(n_jobs=-1)
        case _:
            raise ValueError("Invalid sampler")


def get_combined_sampler(sampler: str):
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
            return SMOTEENN(n_jobs=-1)
        case "SMOTETomek":
            return SMOTETomek(n_jobs=-1)
        case _:
            raise ValueError("Invalid sampler")


# A function to preprocess the data
def preprocess_data(
    train_data: np.ndarray,
    test_data: np.ndarray,
    dataset_name: str,
    _X_train_dict: dict,
    _X_test_dict: dict,
    _y_train_dict: dict,
    _y_test_dict: dict,
) -> None:
    """
    Preprocesses the given train and test data for a specific dataset.

    Args:
        train_data (np.ndarray): The training data array.
        test_data (np.ndarray): The test data array.
        dataset_name (str): The name of the dataset.
        _X_train_dict (dict): The dictionary to store the preprocessed training data.
        _X_test_dict (dict): The dictionary to store the preprocessed test data.
        _y_train_dict (dict): The dictionary to store the training labels.
        _y_test_dict (dict): The dictionary to store the test labels.

    Returns:
        None
    """
    _X_train, _y_train, _X_test, _y_test = (
        train_data[:, :-1],
        train_data[:, -1].astype(int),
        test_data[:, :-1],
        test_data[:, -1].astype(int),
    )
    pca = PCA(n_components=30, random_state=0).fit(_X_train)
    _X_train = pca.transform(_X_train)
    _X_test = pca.transform(_X_test)
    scaler = MinMaxScaler().fit(_X_train)
    _X_train = scaler.transform(_X_train)
    _X_test = scaler.transform(_X_test)
    _X_train_dict[dataset_name] = _X_train
    _X_test_dict[dataset_name] = _X_test
    _y_train_dict[dataset_name] = _y_train
    _y_test_dict[dataset_name] = _y_test


def train_model(
    _type: str,
    _samplers: list[str],
    _result: pd.DataFrame,
    no_sampler: bool = False,
):
    """
    Train a machine learning model.

    Args:
        _type (str): The type of model to train.
        _samplers (list[str]): The list of samplers to use for training.
        _result (pd.DataFrame): The dataframe to store the results.
        no_sampler (bool, optional): If True, train the model without using any sampler. Defaults to False.

    Returns:
        None

    Raises:
        ValueError: If an invalid sampler type is provided.

    """
    if no_sampler:
        # Initialize a list to store the scores
        score = []
        pbar_set = tqdm(dataset_indexes)
        for _dataset_index in pbar_set:
            # Skip the test datasets
            if _dataset_index % 2 == 0:
                continue
            # Update the progress bar
            pbar_set.set_description(f"Training in {dataset_names[_dataset_index]}")
            # Get the name of the current dataset
            dataset_name = dataset_names[_dataset_index]
            # Split the datasets
            X_train, y_train, X_test, y_test = (
                X_train_dict[dataset_name],
                y_train_dict[dataset_name],
                X_test_dict[dataset_name],
                y_test_dict[dataset_name],
            )
            # Initialize the classifier
            clf = RandomForestClassifier(n_jobs=-1, random_state=0)
            # Fit the classifier
            clf.fit(X_train, y_train)
            # Predict the test data
            pred = clf.predict(X_test)
            # Calculate the score
            score.append(geometric_mean_score(y_test, pred))
            # Append the score to the dataframe
        _result.loc[-1] = [_type, "Null", *score, 0]
        _result.index = _result.index + 1
    else:
        _pbar = tqdm(_samplers)
        # Iterate over the samplers
        for sampler in _pbar:
            # Update the progress bar
            _pbar.set_description(f"Training with {sampler}")
            # Initialize a list to store the scores
            score = []
            pbar_set = tqdm(dataset_indexes)
            # Iterate over the datasets
            for _dataset_index in pbar_set:
                # Skip the test datasets
                if _dataset_index % 2 == 0:
                    continue
                # Update the progress bar
                pbar_set.set_description(f"Training in {dataset_names[_dataset_index]}")
                # Get the sampler
                match _type:
                    case "Under-Sampling":
                        _sampler = get_under_sampler(sampler)
                    case "Over-Sampling":
                        _sampler = get_over_sampler(sampler)
                    case "Combined-Sampling":
                        _sampler = get_combined_sampler(sampler)
                    case _:
                        raise ValueError("Invalid sampler type")
                # Get the name of the current dataset
                dataset_name = dataset_names[_dataset_index]
                # Split the datasets
                X_train, y_train, X_test, y_test = (
                    X_train_dict[dataset_name],
                    y_train_dict[dataset_name],
                    X_test_dict[dataset_name],
                    y_test_dict[dataset_name],
                )
                try:
                    # Sample the train data with sampler
                    X_res, y_res = _sampler.fit_resample(X_train, y_train)
                    # Initialize the classifier
                    clf = RandomForestClassifier(n_jobs=-1, random_state=0)
                    # Fit the classifier
                    clf.fit(X_res, y_res)
                    # Predict the test data
                    pred = clf.predict(X_test)
                    # Calculate and record the score
                    score.append(geometric_mean_score(y_test, pred))
                # If the sampling failed
                except RuntimeError as e:
                    score.append(0)
                    print(str(e))
                except Exception as e:
                    score.append(0)
                    print(str(e))
                    logging.exception(e)
                else:
                    print("Successfully sampled the data.")

            # Append the score to the dataframe
            _result.loc[-1] = [_type, sampler, *score, 0]
            _result.index = _result.index + 1


def main():
    """
    Main function that executes the entire program.

    The function iterates over the datasets and preprocesses the data. Then, it initializes a dataframe to store the results and trains models using different sampling methods. The function also calculates the average rank of the results and sorts the results by type. Finally, it initializes a Dash app and displays a table with the results.

    Returns:
        None
    """
    # Iterate over the datasets
    pbar = tqdm(dataset_indexes)
    # Preprocess the data
    for dataset_index in pbar:
        # Skip the test datasets
        if dataset_index % 2 == 0:
            continue
        # Update the progress bar
        pbar.set_description(f"Preprocessing {dataset_names[dataset_index]}")
        preprocess_data(
            data_dict[dataset_names[dataset_index]],
            data_dict[dataset_names[dataset_index - 1]],
            dataset_names[dataset_index],
            X_train_dict,
            X_test_dict,
            y_train_dict,
            y_test_dict,
        )

    # Initialize a dataframe to store the results
    result = pd.DataFrame(
        columns=[
            "Type",
            "Sampler",
            "Cancer",
            "FACS_CD8",
            "PBMC_Batch",
            "PBMC_COVID",
            "cSCC",
            "Average Rank",
        ]
    )

    # Train the models
    train_model("Null", _samplers=[], _result=result, no_sampler=True)
    train_model("Under-Sampling", _samplers=under_samplers, _result=result)
    train_model("Over-Sampling", _samplers=over_samplers, _result=result)
    train_model("Combined-Sampling", _samplers=combined_samplers, _result=result)

    result = average_rank(
        result, ["Cancer", "FACS_CD8", "PBMC_Batch", "PBMC_COVID", "cSCC"]
    )
    # TODO: sort the result with "Type"

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
                        "filter_query": "{{Cancer}}={}".format(
                            _
                        ),  # Highlight the top three cells in "Cancer" column
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
                        "filter_query": "{{cSCC}}={}".format(
                            _
                        ),  # Highlight the top three cells in "cSCC" column
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
                        "filter_query": "{{Average Rank}}={}".format(
                            _
                        ),  # Highlight the top three sampling methods
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


if __name__ == "__main__":
    main()
    pass
