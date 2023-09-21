import os
from typing import Dict

import anndata as ad
import numpy as np
from sklearn.preprocessing import LabelEncoder


def read_h5ad_datasets(directory: str) -> Dict[str, ad.AnnData]:
    """
    Read all train&test files and use a dictionary to store them.
    param directory: the path to the directory of the datasets
    return: a dictionary stores all the datasets
    """
    return {
        os.path.splitext(file)[0]: ad.read_h5ad(os.path.join(root, file))
        for root, _, files in os.walk(directory)
        for file in files
        if file.endswith(".h5ad")
    }


def convert_dataset_dict_to_np_dict(
    dataset_dict: Dict[str, ad.AnnData]
) -> Dict[str, np.ndarray]:
    """
    Transform the data structure from ad.AnnData to np.array for training and testing.
    param dict_: a dictionary stores all the datasets
    return: the same dictionary other than the inner data-structure
    """
    np_dict = {}

    for dataset_name, adata in dataset_dict.items():
        label_encoder = LabelEncoder()
        encoded_values = label_encoder.fit_transform(adata.obs["cell.type"]).reshape(
            -1, 1
        )
        np_dict[dataset_name] = np.concatenate(
            (np.array(adata.X), encoded_values), axis=1
        )

    return dict(sorted(np_dict.items()))


def get_data_dict(directory: str) -> Dict[str, np.ndarray]:
    """
    Get the data_dict from the directory of the datasets
    """
    return convert_dataset_dict_to_np_dict(read_h5ad_datasets(directory))


# from deep_river.classification import RollingClassifier
# from river import metrics, compose, preprocessing
# import torch
#
#
# class MyModule(torch.nn.Module):
#
#     def __init__(self, n_features, hidden_size=1):
#         super().__init__()
#         self.n_features=n_features
#         self.hidden_size = hidden_size
#         self.lstm = torch.nn.LSTM(input_size=n_features, hidden_size=hidden_size,
#         batch_first=False,num_layers=1,bias=False)
#         self.softmax = torch.nn.Softmax(dim=-1)
#
#     def forward(self, X, **kwargs):
#         output, (hn, cn) = self.lstm(X)
#         hn = hn.view(-1, self.lstm.hidden_size)
#         return self.softmax(hn)
#
# metric = metrics.Accuracy()
# optimizer_fn = torch.optim.SGD
#
# model_pipeline = preprocessing.StandardScaler()
# model_pipeline |= RollingClassifier(
#     module=MyModule,
#     loss_fn="cross_entropy",
#     optimizer_fn=torch.optim.SGD,
#     window_size=20,
#     lr=1e-2,
#     append_predict=True,
#     is_class_incremental=True
# )
#
# for x, y in dataset.take(5000):
#     y_pred = model_pipeline.predict_one(x)  # make a prediction
#     metric = metric.update(y, y_pred)  # update the metric
#     model = model_pipeline.learn_one(x, y)  # make the model learn
#
# print(f'Accuracy: {metric.get()}')
