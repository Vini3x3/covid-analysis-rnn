import os

import numpy as np
import pandas as pd
import torch

from loader.DataLoader import read_dataframe
from model.ScalingResidualLstmModel import ScalingResidualLstmModel

WAVE = 1


def create_pd_top(names, data):
    return pd.DataFrame({'name': names, 'data': data}).sort_values(by='data', ascending=False).head(5)


def get_file_path(wave, filename):
    curr_dir = os.getcwd()
    project_dir = curr_dir.split('GitHub')[0]
    analysis_on_covid_dir = os.path.join(project_dir, 'GitHub', 'analysis-on-covid')
    return analysis_on_covid_dir + 'checkpoint_wave' + str(wave) + '/' + filename

# load from checkpoints
best_train_loss = np.load(get_file_path(WAVE, 'best_train_loss.npy'))
best_train_loss = np.sort(best_train_loss)

best_lstm_lag = torch.load(get_file_path(WAVE, 'best_lstm_lag.pt'))

best_fc2_bias = torch.load(get_file_path(WAVE, 'best_fc2_bias.pt'))
best_fc2_weight = torch.load(get_file_path(WAVE, 'best_fc2_weight.pt'))

best_residual_bias = torch.load(get_file_path(WAVE, 'best_residual_bias.pt'))
best_residual_weight = torch.load(get_file_path(WAVE, 'best_residual_weight.pt'))

best_lstm_weight_ih0 = torch.load(get_file_path(WAVE, 'best_lstm_weight_ih0.pt'))
best_lstm_weight_hh0 = torch.load(get_file_path(WAVE, 'best_lstm_weight_hh0.pt'))
best_lstm_bias_ih0 = torch.load(get_file_path(WAVE, 'best_lstm_bias_ih0.pt'))
best_lstm_bias_hh0 = torch.load(get_file_path(WAVE, 'best_lstm_bias_hh0.pt'))


# get target models
### get models with min training loss below 25th percentile
threshold = np.percentile(best_train_loss, 25)
indices = [i for i, x in enumerate(best_train_loss) if x <= threshold]
### filter model which have fc2 both weights are positive
filtered_indices = []
for i in indices:
    weights = list(list(best_fc2_weight[i]))[0]
    if torch.all(weights > 0):
        filtered_indices.append(i)
indices = filtered_indices
print("%d models are selected" % len(indices))



# columns
dataframe_headers = list(read_dataframe('all').columns)[1:]

print("### LSTM section")
print("beyond the first layer, all the input and hidden are 64*64, so we cannot determine which input is important")
print("we only focus on models with minimal training loss below 25th quantile")
print()

layer_0_input_var_abs_weight_above_median = best_lstm_weight_ih0[indices][:, 0:64, :].abs().mean(dim=0).mean(
    dim=0).tolist()
print("layer 0's input gate's weight for each variable for best_train_loss")
print(create_pd_top(dataframe_headers, layer_0_input_var_abs_weight_above_median))
print("this suggests the above variables' input are important prediction ")
print()

layer_0_forget_var_abs_weight_above_median = best_lstm_weight_ih0[indices][:, 64 * 1:64 * 2, :].abs().mean(dim=0).mean(
    dim=0).tolist()
print("layer 0's forget gate's weight for each variable for best_train_loss")
print(create_pd_top(dataframe_headers, layer_0_forget_var_abs_weight_above_median))
print("this suggests the above variables' input are important prediction ")
print()

print("### Residual")
avg_residual_input_weight_abs_sum = best_residual_weight[indices].abs().mean(dim=0).squeeze().tolist()
print("residual layer with higher weight abs sum")
print(create_pd_top(dataframe_headers, avg_residual_input_weight_abs_sum))
print()

print("### FC2 - weighting for lstm layers to residual")
avg_fc2_input_weight_direction = best_fc2_weight[indices].mean(dim=0).squeeze().tolist()
print(avg_fc2_input_weight_direction)
print("left is LSTM model, and right is Residual network")
print()

print("### Lag - weighting for lag being important")
avg_lstm_lag = best_lstm_lag[indices].mean(dim=0).squeeze().tolist()
softmax_lag = np.exp(avg_lstm_lag) / sum(np.exp(avg_lstm_lag))
print(softmax_lag)
print("the softmax of lag importance, which the first element is the first lag")

# example load model
input_dim = len(dataframe_headers)
hidden_dim = 64
num_layers = 4
output_dim = 1
LAG = 15
model = ScalingResidualLstmModel(input_dim, hidden_dim, num_layers, output_dim, LAG)
model_path: str = get_file_path(WAVE, 'checkpoints/model_{:02d}.chk'.format(0))
model.load_state_dict(torch.load(model_path))
