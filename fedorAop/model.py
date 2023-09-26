from torch import nn


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
        self.input_layer = nn.Linear(in_features, in_features)

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
        num_layers: int = 2,
        batch_first: bool = True,
        dropout: float = 0.2,
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

    def __init__(
        self, input_size: int, hidden_size: int, num_layers: int, dropout: float
    ):
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
        n_layers: int = 2,
        dropout: float = 0.5,
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

    # TODO: Add __repr__ method for every class in this module
    def __repr__(self):
        return f"{self.layers}"
