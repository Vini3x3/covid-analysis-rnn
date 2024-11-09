import copy
import os

import numpy as np
import torch

from loader.DataLoader import read_dataframe
from loader.DataTransformer import lag_list
from model.RandomForestLstmModel import RandomForestLstmModel
from model.ScalingResidualLstmModel import ScalingResidualLstmModel


def get_file_path(wave, filename):
    curr_dir = os.getcwd()
    project_dir = curr_dir.split('GitHub')[0]
    analysis_on_covid_dir = os.path.join(project_dir, 'GitHub', 'analysis-on-covid')
    return analysis_on_covid_dir + 'checkpoint_wave' + str(wave) + '/' + filename


LAG = 15
WAVE = 3
REPEAT = 100

# prepare data
sequence = read_dataframe('all').to_numpy()
y_var = np.var(sequence[:, -1])
shifted_sequence = lag_list(sequence, LAG + 1)  # shift into delayed sequences

x_train = shifted_sequence[:, :-1, 1:]  # for each delayed sequence, take all elements except last element
y_train = shifted_sequence[:, -1, -1]  # for each delayed sequence, only take the last element
y_train = y_train.reshape(-1, 1)

x_train = torch.from_numpy(x_train.astype('float64')).type(torch.Tensor)  # convert to tensor
y_train = torch.from_numpy(y_train.astype('int32')).type(torch.Tensor)  # convert to tensor

if WAVE == 1:
    x_train, y_train = x_train[52 - LAG:103 - LAG], y_train[52 - LAG:103 - LAG]
if WAVE == 2:
    x_train, y_train = x_train[160 - LAG:280 - LAG], y_train[160 - LAG:280 - LAG]
elif WAVE == 3:
    x_train, y_train = x_train[280 - LAG:505 - LAG], y_train[280 - LAG:505 - LAG]
elif WAVE == 4:
    x_train, y_train = x_train[757 - LAG:871 - LAG], y_train[757 - LAG:871 - LAG]

performance = np.load(get_file_path(WAVE, 'best_train_loss.npy'))

# build model
input_dim = x_train.shape[-1]
hidden_dim = 64
num_layers = 4
output_dim = 1
LAG = 15

models = []
for i in range(100):
    model = ScalingResidualLstmModel(input_dim, hidden_dim, num_layers, output_dim, LAG)
    model_path: str = get_file_path(WAVE, 'checkpoints/model_{:02d}.chk'.format(i))
    model.load_state_dict(torch.load(model_path))
    models.append(copy.deepcopy(model))

rf = RandomForestLstmModel(models, performance)

y_pred = rf(x_train)

print(torch.nn.functional.mse_loss(y_pred, y_train))
