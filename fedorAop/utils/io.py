import os

import anndata as ad
import numpy as np
from sklearn.preprocessing import LabelEncoder
import json


def read_h5ad_datasets(directory: str) -> dict[str, ad.AnnData]:
    """Read all datasets inside a given folder and use a dictionary to store them

    Parameters
    ----------
    directory : str
        The directory where the Dataset folder is located

    Returns
    -------
    dict[str, ad.AnnData]
        A dictionary whose keys being the name of each dataset, values being the data of each dataset
    """
    return {
        os.path.splitext(file)[0]: ad.read_h5ad(os.path.join(root, file))
        for root, _, files in os.walk(directory)
        for file in files
        if file.endswith(".h5ad")
    }


def h5ad_to_json(directory: str, cell_type_key: str) -> None:
    """Read .h5ad files and write their data, including cell type values, to individual JSON files.

    Parameters
    ----------
    directory : str
        The directory where the .h5ad files are located.
    cell_type_key : str
        The key to access the cell type information in the AnnData object.
    """
    datasets = read_h5ad_datasets(directory)

    for dataset_name, adata in datasets.items():
        data_dict = {"X": adata.X.tolist(), "obs": {cell_type_key: adata.obs[cell_type_key].tolist()}}

        output_file = f"{dataset_name}.json"
        with open(output_file, "w") as json_file:
            json.dump(data_dict, json_file)


def convert_dataset_dict_to_np_dict(dataset_dict: dict[str, ad.AnnData]) -> dict[str, np.ndarray]:
    """
    Convert each dataset in the given dictionary from `ad.AnnData` to `np.ndarray`,
    encode the 'cell_type' stored in `adata.obs['cell.type']` as numeric labels,
    and append the labels to the data.

    Parameters
    ----------
    dataset_dict : dict[str, ad.AnnData]
        A dictionary whose keys are the names of each dataset and values are the data of each dataset.

    Returns
    -------
    dict[str, np.ndarray]
         A dictionary transformed from the input dictionary, with the last column encoded from the 'cell_type' of each observation.
    """
    np_dict = {}

    for dataset_name, adata in dataset_dict.items():
        # Initialize label encoder
        label_encoder = LabelEncoder()

        # Encode 'cell_type' to numeric labels
        encoded_values = label_encoder.fit_transform(adata.obs["cell.type"])

        # Append encoded values to the data
        np_dict[dataset_name] = np.concatenate((np.array(adata.X), encoded_values[:, None]), axis=1)

    # Sort and return the transformed dictionary
    return dict(sorted(np_dict.items()))


def get_data_dict(directory: str) -> dict[str, np.ndarray]:
    """
    Generates a dictionary of data arrays from the datasets stored in the specified directory.

    :param directory: The directory where the datasets are stored.
    :type directory: str
    :return: A dictionary mapping dataset names to NumPy arrays.
    :rtype: dict[str, np.ndarray]
    """
    return convert_dataset_dict_to_np_dict(read_h5ad_datasets(directory))
