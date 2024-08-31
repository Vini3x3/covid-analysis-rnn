import copy

import numpy as np
import torch

from loader.DataLoader import read_sequence
from loader.DataTransformer import lag_list, diff_matrix, normalize_matrix, normalize, diff
from model.FcLstmModel import FcLstmModel


def create_sequence(seq, lag):
    seq = seq.reshape(-1, 1)
    shifted_sequence = lag_list(seq, lag)  # shift into delayed sequences

    x_train = shifted_sequence[:, :-1, :]  # for each delayed sequence, take all elements except last element
    y_train = shifted_sequence[:, -1, :]  # for each delayed sequence, only take the last element

    x_train = torch.from_numpy(x_train).type(torch.Tensor)  # convert to tensor
    y_train = torch.from_numpy(y_train).type(torch.Tensor)  # convert to tensor

    return x_train, y_train


# data parameter
LAG = 16

# prepare data
sequence = read_sequence('case')

wave_1 = sequence[43:88]
wave_2 = sequence[146:265]
wave_3 = sequence[266:490]
wave_4 = sequence[702:723]

wave_2 = diff(wave_2)
wave_2 = normalize(wave_2)

wave_1_x, wave_1_y = create_sequence(wave_1, LAG)
wave_2_x, wave_2_y = create_sequence(wave_2, LAG)
wave_3_x, wave_3_y = create_sequence(wave_3, LAG)
wave_4_x, wave_4_y = create_sequence(wave_4, LAG)

# build model
input_dim = 1
hidden_dim = 64
num_layers = 2
output_dim = 1

model = FcLstmModel(input_dim, hidden_dim, num_layers, output_dim, LAG - 1, 0, 0)
# model = LstmModel(input_dim, hidden_dim, num_layers, output_dim)

# train
num_epochs = 2_000
loss_fn = torch.nn.MSELoss()
optimiser = torch.optim.Adam(model.parameters(), lr=0.01)
min_loss = np.inf
best_model_state = None
model.train()

TRAIN_RANGE = int(np.floor(wave_2_y.shape[0] * 2 / 3))

y_var = np.var(wave_2[:TRAIN_RANGE])
for epoch in range(1, num_epochs + 1):
    y_pred = model(wave_2_x[:TRAIN_RANGE])
    loss = loss_fn(y_pred, wave_2_y[:TRAIN_RANGE])
    if epoch % 100 == 0:
        print("Epoch: %d | MSE: %.2E | RRSE: %.2E" % (epoch, loss.item(), np.sqrt(loss.item() / y_var)))
    if min_loss > loss.item():
        best_model_state = copy.deepcopy(model.state_dict())
        min_loss = loss.item()
    optimiser.zero_grad()
    loss.backward()
    optimiser.step()

# y_var = np.var(wave_2)
# for epoch in range(1, num_epochs + 1):
#     y_pred = model(wave_2_x)
#     loss = loss_fn(y_pred, wave_2_y)
#     if epoch % 100 == 0:
#         print("Epoch: %d | MSE: %.2E | RRSE: %.2E" % (epoch, loss.item(), np.sqrt(loss.item() / y_var)))
#     if min_loss > loss.item():
#         best_model_state = copy.deepcopy(model.state_dict())
#         min_loss = loss.item()
#     optimiser.zero_grad()
#     loss.backward()
#     optimiser.step()

# predict
model.eval()
# model.load_state_dict(best_model_state)
x_test = wave_2_x[TRAIN_RANGE].reshape(1, LAG - 1, 1)
y_pred = []
# prediction_range = wave_2_x.shape[0] - TRAIN_RANGE
prediction_range = 7
for _ in range(prediction_range):
    _ = model(x_test)
    x_test = torch.cat((x_test[0][1:], _), dim=0)
    x_test = x_test.reshape(1, LAG - 1, 1)
    _2 = _.detach().numpy()  # revert from tensor
    _2 = _2.reshape(-1)  # reshape back to normal list
    y_pred.append(_2)

y_pred = np.array(y_pred).reshape(-1)  # reshape back to normal list
print("sample prediction:  ", y_pred)

y_train_sample = wave_2_y[TRAIN_RANGE:].detach().numpy().reshape(-1)[:prediction_range]
print("sample true result: ", y_train_sample)

mse = sum((y_train_sample - y_pred) ** 2) / len(y_pred)
rmse = mse / np.var(y_train_sample)
print("TEST | MSE: %.2E | RRSE: %.2E" % (mse, rmse))
