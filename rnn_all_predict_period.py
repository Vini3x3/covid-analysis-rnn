import copy

import numpy as np
import torch

from loader.DataLoader import read_dataframe
from loader.DataTransformer import lag_list, transform_matrix
from model.CnnLstmModel import CnnLstmModel


def create_sequence(seq, lag):
    shifted_sequence = lag_list(seq, lag)  # shift into delayed sequences

    x_train = shifted_sequence[:, :-1, :]  # for each delayed sequence, take all elements except last element
    y_train = shifted_sequence[:, -1, :]  # for each delayed sequence, only take the last element

    x_train = torch.from_numpy(x_train.astype('float64')).type(torch.Tensor)  # convert to tensor
    y_train = torch.from_numpy(y_train.astype('float64')).type(torch.Tensor)  # convert to tensor

    return x_train, y_train


# data parameter
LAG = 15

# prepare data
sequence = read_dataframe('all').to_numpy()
sequence = sequence[:, 1:]
sequence = transform_matrix(sequence, 'DIFF')
sequence = transform_matrix(sequence, 'NORM')
sequence[np.isnan(sequence)] = 0 # fill na - there is a column which are all 0

shifted_sequence = lag_list(sequence, LAG + 1)  # shift into delayed sequences

x_train = shifted_sequence[:, :-1, 1:]  # for each delayed sequence, take all elements except last element
y_train = shifted_sequence[:, -1, -1]  # for each delayed sequence, only take the last element
y_train = y_train.reshape(-1, 1)

x_train = torch.from_numpy(x_train.astype('float64')).type(torch.Tensor)  # convert to tensor
y_train = torch.from_numpy(y_train.astype('float64')).type(torch.Tensor)  # convert to tensor

wave_1_x, wave_1_y = x_train[43 -LAG:88 -LAG], y_train[43 -LAG:88 -LAG]
wave_2_x, wave_2_y = x_train[146 -LAG:265 -LAG], y_train[146 -LAG:265 -LAG]
wave_3_x, wave_3_y = x_train[266 -LAG:490 -LAG], y_train[266 -LAG:490 -LAG]
wave_4_x, wave_4_y = x_train[702 -LAG:723 -LAG], y_train[702 -LAG:723 -LAG]

# build model
input_dim = x_train.shape[-1]
hidden_dim = 64
num_layers = 2
output_dim = 1

model = CnnLstmModel(input_dim, LAG + 1)

# train
num_epochs = 2_000
loss_fn = torch.nn.MSELoss()
optimiser = torch.optim.Adam(model.parameters(), lr=0.01)
min_loss = np.inf
best_model_state = None
model.train()

TRAIN_RANGE = int(np.floor(wave_2_y.shape[0] * 2 / 3))

y_var = np.var(wave_2_y.numpy().reshape(-1)[:TRAIN_RANGE])
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

# predict
model.eval()
model.load_state_dict(best_model_state)
x_test = wave_2_x[TRAIN_RANGE].reshape(1,LAG, wave_2_x.shape[2])
y_pred = []
# prediction_range = wave_2_x.shape[0] - TRAIN_RANGE
prediction_range = 7
for _ in range(prediction_range):
    _ = model(x_test)
    # the prediction cannot fit into multi dimension data!!!
    x_test = torch.cat((x_test[0][1:], _.item()), dim=0)
    x_test = x_test.reshape(1,LAG, wave_2_x.shape[2])
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
