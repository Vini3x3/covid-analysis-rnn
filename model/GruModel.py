import torch
import torch.nn as nn


class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, bidirectional=False):
        super(GRUModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, bidirectional=bidirectional)
        self.bidirectional = bidirectional
        dim = hidden_size
        if bidirectional:
            dim *= 2
        self.fc = nn.Linear(dim, output_size)

    def forward(self, x):
        dim = self.num_layers
        if self.bidirectional:
            dim *= 2
        h0 = torch.zeros(dim, x.size(0), self.hidden_size)

        out, _ = self.gru(x, h0)
        out = self.fc(out[:, -1, :])  # Get the last time step output and pass it through a fully connected layer
        return out