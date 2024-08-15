import numpy as np
import pandas as pd
import torch

from lib.covid_module import get_date_count
from loader.DataLoader import read_sequence
from loader.DataTransformer import lag_list, moving_average
from model.LstmModel import LstmModel

# script parameter
# MODE: MA (moving average), D1(lag 1 degree), DMA(decaying moving average) or default no change
MODE = 'MA'


# prepare data
def transform_sequence(input_sequence: np.ndarray, mode: str = '') -> np.ndarray:
    if mode == 'MA':
        return moving_average(input_sequence, 14)
    elif mode == 'DMA':
        return moving_average(input_sequence, 14, 0.95)
    elif mode == 'D1':
        return np.diff(input_sequence)
    else:
        return input_sequence


# df_infected = pd.read_csv("data/covid/covid_hk_case_std.csv")
# df_infected['report_date'] = pd.to_datetime(df_infected['report_date'], format='%Y%m%d')  # convert to datetime type
# print(df_infected.shape)
#
# df_count = get_date_count(df_infected, 'report_date', '%Y%m%d')
# print(df_count.head())
# print(df_count.shape)

sequence = read_sequence('case')
sequence = transform_sequence(sequence, MODE)
sequence = sequence.reshape(-1, 1)
shifted_sequence = lag_list(sequence, 16)  # shift into delayed sequences

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
num_epochs = 3_000
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
y_pred = model(x_train[:5])
y_pred = y_pred.detach().numpy()  # revert from tensor
y_pred = y_pred.reshape(-1)  # reshape back to normal list
print("sample prediction:  ", y_pred)

y_train_sample = y_train[:5].detach().numpy().reshape(-1)
print("sample true result: ", y_train_sample)

# verify
y_pred_round = [round(p) for p in y_pred]
y_train_round = [round(p) for p in y_train_sample]

assert (y_pred_round == y_train_round)
