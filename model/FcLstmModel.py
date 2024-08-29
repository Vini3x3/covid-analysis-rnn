import torch
import torch.nn as nn


class WeightedOutputLayer(nn.Module):
    def __init__(self, input_size):
        super(WeightedOutputLayer, self).__init__()
        self.weight_vector = nn.Parameter(torch.ones(input_size))

    def forward(self, input_tensor):
        return input_tensor * self.weight_vector


class FcLstmModel(nn.Module):
    def __init__(self, input_size, lstm_hidden_size, lstm_num_layers, output_size, seq_len, lstm_dropout, nn_dropout):
        super(FcLstmModel, self).__init__()

        # Importance Weighting
        # self.fc0 = nn.Linear(input_size, input_size)
        # with torch.no_grad():
        #     self.fc0.weight.fill_(1.0)
        self.fc0 = WeightedOutputLayer(input_size)

        # LSTM Encoding
        self.lstm = nn.LSTM(input_size, lstm_hidden_size, lstm_num_layers, batch_first=True, dropout=lstm_dropout)

        # Neural Network
        self.fc1 = nn.Linear(lstm_hidden_size * seq_len, output_size)
        self.dropout1 = nn.Dropout(nn_dropout)

    def forward(self, x):
        out = self.fc0(x)
        out, _ = self.lstm(out)
        out = out.reshape(x.shape[0], -1) # flatten to pass full sequence
        out = self.fc1(out)
        out = self.dropout1(out)

        return out