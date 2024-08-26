import torch
import torch.nn as nn

class LstmModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LstmModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])  # Get the last time step output and pass it through a fully connected layer
        return out