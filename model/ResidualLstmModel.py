import torch
import torch.nn as nn

class ResidualLstmModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0):
        super(ResidualLstmModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)
        self.residual = nn.Linear(input_size, output_size)
        self.fc2 = nn.Linear(2, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        # out += self.residual(x[:, -1, :])
        residual = self.residual(x[:, -1, :])

        out = self.fc2(torch.cat((out, residual), dim=1))
        return out