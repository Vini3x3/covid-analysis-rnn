import torch
from torch import nn


class TemporalWeightingLayer(nn.Module):
    def __init__(self, lag):
        super(TemporalWeightingLayer, self).__init__()
        self.lag_importance = nn.Parameter(torch.full((lag,), 10.0))

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, features, lag)
        Returns:
            Amplified tensor of the same shape (batch_size, features, lag)
        """
        weights = torch.sigmoid(self.lag_importance)
        amplified_x = x * weights.view(1, -1, 1)
        return amplified_x


class ScalingResidualLstmModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, lag, dropout=0):
        super(ScalingResidualLstmModel, self).__init__()
        self.scaling = TemporalWeightingLayer(lag)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)
        self.residual = nn.Linear(input_size, output_size)
        self.fc2 = nn.Linear(2, output_size)

    def forward(self, x):
        out = self.scaling(x)
        out, _ = self.lstm(out)
        out = self.fc(out[:, -1, :])
        residual = self.residual(x[:, -1, :])  # last timestamp of output only

        out = self.fc2(torch.cat((out, residual), dim=1))
        return out