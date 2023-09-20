from torch import nn


class InputLayer(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.input_layer = nn.Linear(in_features, in_features)

    def forward(self, X):
        X = self.input_layer(X)
        return X


class EmbedLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.embedding = nn.Linear(in_features, out_features)

    def forward(self, X):
        X = self.embedding(X)
        return X


class RNNLayers(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
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
    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super().__init__()
        self.layers = nn.LSTM(input_size, hidden_size, num_layers, dropout)

    def forward(self, X):
        X = X.unsqueeze(1)
        X, _ = self.layers(X)
        X = X[:, -1, :]
        return X


class MLP(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
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

    def buildMlp(self):
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


class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList()

    def buildLayers(self, *layers: nn.Module):
        for layer in layers:
            self.layers.append(layer)

    def forward(self, X):
        for layer in self.layers:
            X = layer(X)

        return X

    # modify this
    def __repr__(self):
        return f"{self.layers}"
