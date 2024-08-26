import torch.nn as nn

class CnnLstmModel(nn.Module):
    def __init__(self, num_features, num_timesteps):
        super(CnnLstmModel, self).__init__()

        # 1D Convolutional layers
        self.conv1 = nn.Conv1d(in_channels=num_features, out_channels=64, kernel_size=3)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3)

        # LSTM layer
        self.lstm = nn.LSTM(input_size=64, hidden_size=50, num_layers=2, batch_first=True)

        # Fully connected layer
        self.fc1 = nn.Linear(50 * (num_timesteps - 5), 100)  # Adjust for output size after CNN layers
        self.fc2 = nn.Linear(100, 1)

        self.dropout = nn.Dropout(0.2)
        self.relu = nn.ReLU()

    def forward(self, x):
        # Input shape: (batch_size, num_timesteps, num_features)
        x = x.permute(0, 2, 1)  # Change shape to (batch_size, num_features, num_timesteps) for Conv1d

        # Apply Conv1D layers
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))

        # Permute back to shape (batch_size, num_timesteps, num_features) for LSTM
        x = x.permute(0, 2, 1)

        # Apply LSTM layers
        x, _ = self.lstm(x)

        # Flatten the output from LSTM
        x = x.contiguous().view(x.size(0), -1)

        # Apply fully connected layers
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x