import copy
import os

import numpy as np
import pandas as pd
import torch

from lib.uilt_module import CurveLoss
from loader.DataLoader import read_dataframe
from loader.DataTransformer import lag_list
from model.ScalingResidualLstmModel import ScalingResidualLstmModel

# script parameter
LAG = 15
WAVE = 4
REPEAT = 100


def get_file_path(wave, filename):
    curr_dir = os.getcwd()
    project_dir = curr_dir.split('GitHub')[0]
    analysis_on_covid_dir = os.path.join(project_dir, 'GitHub', 'analysis-on-covid')
    return analysis_on_covid_dir + 'checkpoint_wave' + str(wave) + '/' + filename
    # return 'checkpoint_wave' + str(wave) + '/' + filename


def get_wave_period(wave: int):
    if wave == 1:
        return 52, 103
    elif wave == 2:
        return 160, 280
    elif wave == 3:
        return 280, 505
    elif wave == 4:
        return 757, 871


def std_column(df: pd.DataFrame, column_name: str) -> pd.DataFrame:
    if (df[column_name] == 0).all():
        return df[column_name]
    return (df[column_name] - df[column_name].mean()) / df[column_name].std()


print("WAVE %d" % WAVE)

# prepare data
wave_start, wave_end = get_wave_period(WAVE)

sequence = read_dataframe('all')
sequence = sequence[wave_start - LAG: wave_end]

numeric_columns = ['avg_temp', 'avg_humid', 'sum', 'count']
for _ in numeric_columns:
    sequence.loc[:, _] = std_column(sequence, _)

shifted_sequence = lag_list(sequence, LAG + 1)  # shift into delayed sequences

x_train = shifted_sequence[:, :-1, 1:]  # for each delayed sequence, take all elements except last element
y_train = shifted_sequence[:, -1, -1]  # for each delayed sequence, only take the last element
y_train = y_train.reshape(-1, 1)

x_train = torch.from_numpy(x_train.astype('float64')).type(torch.Tensor)  # convert to tensor
y_train = torch.from_numpy(y_train.astype('float64')).type(torch.Tensor)  # convert to tensor

# build model
input_dim = x_train.shape[-1]
hidden_dim = 64
num_layers = 4
output_dim = 1

### looping for repetitive training
best_models = []
best_training_loss = []

best_residual = []
best_fc2 = []

best_lstm_weight_ih0 = []
best_lstm_weight_hh0 = []
best_lstm_bias_ih0 = []
best_lstm_bias_hh0 = []

best_lstm_lag = []

for _ in range(REPEAT):
    best_model_state = None
    best_model_train_loss = np.inf
    model = ScalingResidualLstmModel(input_dim, hidden_dim, num_layers, output_dim, LAG)

    # train
    num_epochs = 3_000
    loss_fn = CurveLoss(model, 1e-5)
    optimiser = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.train()
    for epoch in range(1, num_epochs + 1):
        y_pred = model(x_train)
        loss = loss_fn(y_pred, y_train)
        if best_model_train_loss > loss.item():
            best_model_state = copy.deepcopy(model.state_dict())
            best_model_train_loss = loss.item()
        if epoch % 100 == 0:
            print("Epoch: %d | LOSS: %.2E" % (epoch, loss.item()))
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

    # test
    model.load_state_dict(best_model_state)

    best_models.append(best_model_state)
    best_training_loss.append(best_model_train_loss)

    best_fc2.append(model.fc2)
    best_residual.append(model.residual)

    best_lstm_weight_ih0.append(model.lstm.weight_ih_l0.data)
    best_lstm_weight_hh0.append(model.lstm.weight_hh_l0.data)
    best_lstm_bias_ih0.append(model.lstm.bias_ih_l0.data)
    best_lstm_bias_hh0.append(model.lstm.bias_hh_l0.data)

    best_lstm_lag.append(model.scaling.lag_importance.data)

### save files

torch.save(torch.stack([_.bias.data for _ in best_fc2], dim=0), get_file_path(WAVE, "best_fc2_bias.pt"))
torch.save(torch.stack([_.weight.data for _ in best_fc2], dim=0), get_file_path(WAVE, "best_fc2_weight.pt"))

torch.save(torch.stack([_.bias.data for _ in best_residual], dim=0), get_file_path(WAVE, "best_residual_bias.pt"))
torch.save(torch.stack([_.weight.data for _ in best_residual], dim=0), get_file_path(WAVE, "best_residual_weight.pt"))

torch.save(torch.stack(best_lstm_weight_ih0, dim=0), get_file_path(WAVE, "best_lstm_weight_ih0.pt"))
torch.save(torch.stack(best_lstm_weight_hh0, dim=0), get_file_path(WAVE, "best_lstm_weight_hh0.pt"))
torch.save(torch.stack(best_lstm_bias_ih0, dim=0), get_file_path(WAVE, "best_lstm_bias_ih0.pt"))
torch.save(torch.stack(best_lstm_bias_hh0, dim=0), get_file_path(WAVE, "best_lstm_bias_hh0.pt"))

torch.save(torch.stack(best_lstm_lag, dim=0), get_file_path(WAVE, "best_lstm_lag.pt"))

np.save(get_file_path(WAVE, 'best_train_loss.npy'), np.array(best_training_loss))

if not os.path.exists(get_file_path(WAVE, 'checkpoints')):
    os.mkdir(get_file_path(WAVE, 'checkpoints'))

for i in range(len(best_models)):
    model_checkpoint_path = get_file_path(WAVE, 'checkpoints/model_' + '{:02d}'.format(i) + '.chk')
    torch.save(best_models[i], model_checkpoint_path)
