import numpy as np
import torch

from loader.DataTransformer import delay_list
from model.LstmModel import LstmModel

"""
Scenario: 
To train a RNN model that can guess the next number, 
e.g. given [3,4,5] it can predict 6 
"""

# prepare data
sequence = list(range(10))
reshaped_sequence = np.array(sequence).reshape(-1, 1)  # reshape into individual list
shifted_sequence = delay_list(reshaped_sequence, 4)  # shift into delayed sequences

x_train = shifted_sequence[:, :-1, :]  # for each delayed sequence, take all elements except last element
y_train = shifted_sequence[:, -1, :]  # for each delayed sequence, only take the last element

x_train = torch.from_numpy(x_train).type(torch.Tensor)  # convert to tensor
y_train = torch.from_numpy(y_train).type(torch.Tensor)  # convert to tensor

# build model
input_dim = 1
hidden_dim = 32
num_layers = 2
output_dim = 1

model = LstmModel(input_dim, hidden_dim, num_layers, output_dim)

# train
num_epochs = 500
loss_fn = torch.nn.MSELoss()
optimiser = torch.optim.Adam(model.parameters(), lr=0.01)
model.train()
for epoch in range(1, num_epochs + 1):
    y_pred = model(x_train)
    loss = loss_fn(y_pred, y_train)
    if epoch % 100 == 0:
        print("Epoch: %d | MSE: %.2E" % (epoch, loss.item()))
    optimiser.zero_grad()
    loss.backward()
    optimiser.step()

# test
model.eval()
y_pred = model(x_train)
y_pred = y_pred.detach().numpy()  # revert from tensor
y_pred = y_pred.reshape(-1)  # reshape back to normal list
print("sample prediction:  ", y_pred)
print("sample true result: ", [3, 4, 5, 6, 7, 8, 9])

# verify
y_pred_round = [round(p) for p in y_pred]
assert (y_pred_round == [3, 4, 5, 6, 7, 8, 9])
