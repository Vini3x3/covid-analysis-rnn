import numpy as np
import pandas as pd
import torch
from loader.DataLoader import read_dataframe


WAVE = 4
CHECKPOINT_DIR = 'checkpoint_wave' + str(WAVE)

def create_pd_top(names, data):
    return pd.DataFrame({'name':names,'data':data}).sort_values(by='data', ascending=False).head(10)

best_train_loss = np.load(CHECKPOINT_DIR+'/best_train_loss.npy')
threshold = np.percentile(best_train_loss, 25)
indices = [i for i, x in enumerate(best_train_loss) if x <= threshold]

best_fc2_bias = torch.load(CHECKPOINT_DIR+'/best_fc2_bias.pt')
best_fc2_weight = torch.load(CHECKPOINT_DIR+'/best_fc2_weight.pt')

best_residual_bias = torch.load(CHECKPOINT_DIR+'/best_residual_bias.pt')
best_residual_weight = torch.load(CHECKPOINT_DIR+'/best_residual_weight.pt')

best_lstm_weight_ih0 = torch.load(CHECKPOINT_DIR+'/best_lstm_weight_ih0.pt')
best_lstm_weight_hh0 = torch.load(CHECKPOINT_DIR+'/best_lstm_weight_hh0.pt')
best_lstm_bias_ih0 = torch.load(CHECKPOINT_DIR+'/best_lstm_bias_ih0.pt')
best_lstm_bias_hh0 = torch.load(CHECKPOINT_DIR+'/best_lstm_bias_hh0.pt')

# columns
dataframe_headers = list(read_dataframe('all').columns)[1:]

print("### LSTM section")
print("beyond the first layer, all the input and hidden are 64*64, so we cannot determine which input is important")
print()

layer_0_input_var_abs_weight_above_median = best_lstm_weight_ih0[indices][:, 0:64, :].abs().mean(dim=0).mean(dim=0).tolist()
print("layer 0's input gate's weight for each variable for above median best_train_loss")
print(create_pd_top(dataframe_headers, layer_0_input_var_abs_weight_above_median))
print("this suggests the above variables' input are important prediction ")
print()

layer_0_forget_var_abs_weight_above_median = best_lstm_weight_ih0[indices][:, 64*1:64*2, :].abs().mean(dim=0).mean(dim=0).tolist()
print("layer 0's forget gate's weight for each variable for above median best_train_loss")
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
print("for low training loss model, the residual layer has is opposite helpful to the output")
print()

