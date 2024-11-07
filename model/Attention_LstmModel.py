import torch
import torch.nn as nn

class Attention(nn.Module):
    def __init__(self, embed_size):
        super(Attention, self).__init__()
        self.embed_size = embed_size
        
        self.query = nn.Linear(embed_size, embed_size)
        self.key = nn.Linear(embed_size, embed_size)
        self.value = nn.Linear(embed_size, embed_size)

    def forward(self, x, mask=None):
        # Transform inputs to query, key, value
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        # Calculate attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.embed_size, dtype=torch.float32))

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        weights = nn.functional.softmax(scores, dim=-1)

        # Compute weighted input
        output = torch.matmul(weights, V)
        
        return output, weights

class Attention_ResidualLstmModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, lag_nbr, dropout=0):
        super(Attention_ResidualLstmModel, self).__init__()
        self.attention = Attention(lag_nbr)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)
        self.residual = nn.Linear(input_size, output_size)
        self.fc2 = nn.Linear(2, output_size)

    def forward(self, x):
        attentioned_x, att_weights = self.attention(x.transpose(-2, -1))  # Apply feature attention
        out, _ = self.lstm(attentioned_x.transpose(-2, -1))
        out = self.fc(out[:, -1, :])
        residual = self.residual(x[:, -1, :]) # last timestamp of output only

        out = self.fc2(torch.cat((out, residual), dim=1))
        return out

# class Attention(nn.Module):
#     def __init__(self):
#         super(Attention, self).__init__()

#     def forward(self, x):
#         # Calculate attention scores
#         attention_scores = torch.bmm(x, x.transpose(1, 2))
#         attention_weights = torch.softmax(attention_scores, dim=-1)  # Softmax over the last dimension
        
#         # Compute weighted input
#         weighted_input = torch.bmm(attention_weights, x)  # Weighted sum
#         return weighted_input, attention_weights  # Return weighted input and weights
        
# class Attention(nn.Module):
#     def __init__(self, input_dim):
#         super(Attention, self).__init__()
#         self.query_linear = nn.Linear(input_dim, input_dim)
#         self.key_linear = nn.Linear(input_dim, input_dim)
#         self.value_linear = nn.Linear(input_dim, input_dim)

#     def forward(self, x):
#         # Transform inputs to query, key, value
#         Q = self.query_linear(x)
#         K = self.key_linear(x)
#         V = self.value_linear(x)

#         # Calculate attention scores
#         attention_scores = torch.bmm(Q, K.transpose(1, 2))
#         attention_weights = torch.softmax(attention_scores, dim=-1)

#         # Compute weighted input
#         weighted_input = torch.bmm(attention_weights, V)
#         return weighted_input, attention_weights

# class Attention(nn.Module):
#     def __init__(self, input_size):
#         super(Attention, self).__init__()
#         self.input_size = input_size
        
#         self.query = nn.Linear(input_size, input_size)
#         self.key = nn.Linear(input_size, input_size)
#         self.value = nn.Linear(input_size, input_size)

#     def forward(self, x, mask=None):
#         # Transform inputs to query, key, value
#         Q = self.query(x)
#         K = self.key(x)
#         V = self.value(x)

#         # Calculate attention scores
#         scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.input_size, dtype=torch.float32))

#         if mask is not None:
#             scores = scores.masked_fill(mask == 0, -1e9)
        
#         weights = nn.functional.softmax(scores, dim=-1)

#         # Compute weighted input
#         output = torch.matmul(weights, V)
        
#         return output, weights
