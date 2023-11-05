"""Loss functions and Layers for Neural Network"""

import torch
import torch.nn.functional as F
from torch import nn

from fedorAop.config import (
    FOCAL_LOSS_ALPHA,
    FOCAL_LOSS_GAMMA,
    RNN_NUM_LAYERS,
    RNN_DROPOUT_RATE,
    MLP_DROPOUT_RATE,
    MLP_NUM_LAYERS,
)


class FocalLoss(nn.Module):
    def __init__(self, alpha=FOCAL_LOSS_ALPHA, gamma=FOCAL_LOSS_GAMMA, logits=True, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce

    def forward(self, inputs, targets):
        if self.logits:
            BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduce=False)
        else:
            BCE_loss = F.binary_cross_entropy(inputs, targets, reduce=False)

        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss


class WeightedCrossEntropyLoss(nn.Module):
    def __init__(self, weights):
        super(WeightedCrossEntropyLoss, self).__init__()
        self.weights = weights
        self.loss_fn = nn.CrossEntropyLoss(weight=self.weights, reduction="mean")

    def forward(self, inputs, targets):
        return self.loss_fn(inputs, targets)


class InputLayer(nn.Module):
    """Construct thr input layer for the neural network
    Inherits from torch.nn.Module

    Parameters:
    -----------
    in_features: int
        The number of features of the input data

    Attributes:
    -----------
    input_layer: nn.Linear
        The input layer for the neural network

    Usages:
    ------
    inputLayer = InputLayer(in_features)
    """

    def __init__(self, in_features: int):
        super().__init__()
        self.in_features = in_features
        self.input_layer = nn.Linear(self.in_features, self.in_features)

    def forward(self, X):
        X = self.input_layer(X)
        return X


class EmbedLayer(nn.Module):
    """Construct a layer for dimensionality reduction

    Parameters:
    ----------
    in_features: int
        The number of input features
    out_features: int
        The number of the targeted output features

    Attributes:
    ----------
    embedding:  nn.Linear
        The embedding layer for the neural network

    Usages:
    ------
    embedLayer = EmbedLayer(in_features, out_features)
    """

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.embedding = nn.Linear(in_features, out_features)

    def forward(self, X):
        X = self.embedding(X)
        return X


class RNNLayers(nn.Module):
    """NotImplemented"""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = RNN_NUM_LAYERS,
        batch_first: bool = True,
        dropout: float = RNN_DROPOUT_RATE,
    ):
        super().__init__()
        self.layers = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=batch_first,
        )

    def forward(self, X):
        X = X.unsqueeze(1)
        X, _ = self.layers(X)
        X = X[:, -1, :]
        return X


class LSTMLayers(nn.Module):
    """NotImplemented"""

    def __init__(self, input_size: int, hidden_size: int, num_layers: int, dropout: float):
        super().__init__()
        self.layers = nn.LSTM(input_size, hidden_size, num_layers, dropout)

    def forward(self, X):
        X = X.unsqueeze(1)
        X, _ = self.layers(X)
        X = X[:, -1, :]
        return X


class CNNLayers(nn.Module):
    """NotImplemented"""

    pass


class MLP(nn.Module):
    """Construct the MLP layers

    Parameters:
    ----------
    in_features: int
        The number of input features
    out_features: int
        The number of targets of the classification, or the number of output features
    n_layers: int, optional
        The number of layers, should be >= 2, by default 2
    dropout: float, optional
        The dropout rate for the Dropout module, by default .5
    softmax: bool, optional
        Whether to softmax the output of the MLP, by default True

    Attributes:
    ----------
    in_features: int
        The number of input features
    out_features: int
        The number of targets of the classification, or the number of output features
    n_layers: int, optional
        The number of layers, should be >= 2, by default 2
    dropout: float, optional
        The dropout rate for the Dropout module, by default .5
    softmax: bool, optional
        Whether to softmax the output of the MLP, by default True
    _inter_features: int, protected
        The number of features fed into interlayers of MLP,
        self._inter_features = (self.in_features + self.out_features) * 2 // 3
    layers: nn.ModuleList
        List every layer of the MLP

    Usages:
    ------
    mlpLayers = MLP(in_features, out_features, n_layers, dropout, softmax)
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        n_layers: int = MLP_NUM_LAYERS,
        dropout: float = MLP_DROPOUT_RATE,
        softmax: bool = True,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.n_layers = n_layers
        self.dropout = dropout
        self.softmax = softmax
        self._inter_features = (self.in_features + self.out_features) * 2 // 3
        self.layers = nn.ModuleList([nn.Linear(in_features, self._inter_features)])
        self.buildMlp()

        try:
            assert self.n_layers >= 2
        except AssertionError as e:
            e.add_note("Use EmbedLayer instead if you want to deploy a 1-layer MLP")
            raise

    def buildMlp(self):
        """Build the MLP"""
        for _ in range(self.n_layers - 2):
            self.layers.append(nn.Linear(self._inter_features, self._inter_features))
            self.layers.append(nn.BatchNorm1d(self._inter_features))
            self.layers.append(nn.ELU())
            self.layers.append(nn.Dropout(self.dropout))

        self.layers.append(nn.Linear(self._inter_features, self.out_features))
        if self.softmax:
            self.layers.append(nn.Softmax(dim=1))
        else:
            self.layers.append(nn.BatchNorm1d(self.out_features))
            self.layers.append(nn.ELU())

    def forward(self, X):
        for layer in self.layers:
            X = layer(X)

        return X


class MLPMixer(nn.Module):
    """NotImplemented"""

    pass


class NeuralNetwork(nn.Module):
    """Construct the Neural Network for the project

    Attributes:
    ----------
    layers: nn.ModuleList
        The layers of the neural network

    Usages:
    ------
    model = NeuralNetwork()
    model.buildLayers(inputLayer, embedLayer, mlpLayers, *layers)
    """

    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList()

    def buildLayers(self, *layers: nn.Module):
        """Build the neural network with the input layers

        Parameters:
        ----------
        layers: nn.Module
            The layers to be put into the neural network
        """
        for layer in layers:
            self.layers.append(layer)

    def forward(self, X):
        for layer in self.layers:
            X = layer(X)

        return X

    def __repr__(self):
        return f"{self.layers}"


class DRLayers(nn.Module):
    """Reduce the dimensionality of the data"""

    def __init__(self, input_layer: nn.Module, mlp_layers: MLP):
        super().__init__()
        self.input_layer = input_layer
        self.mlp = mlp_layers
        self.out_features = self.mlp.out_features

    def forward(self, X):
        X = self.input_layer(X)
        X = self.mlp.forward(X)
        return X

    def frozen(self):
        for param in self.parameters():
            param.requires_grad = False
