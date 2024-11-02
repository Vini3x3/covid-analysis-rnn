import numpy as np
import pandas as pd
import torch
from loader.DataLoader import read_dataframe



def create_pd_top(names, data):
    return pd.DataFrame({'name':names,'data':data}).sort_values(by='data', ascending=False).head(10)

best_train_loss = np.load('checkpoint_all/best_train_loss.npy')
threshold = np.percentile(best_train_loss, 25)
indices = [i for i, x in enumerate(best_train_loss) if x <= threshold]


best_fc1_bias = torch.load('checkpoint_all/best_fc1_bias.pt')
best_fc1_weight = torch.load('checkpoint_all/best_fc1_weight.pt')

best_fc2_bias = torch.load('checkpoint_all/best_fc2_bias.pt')
best_fc2_weight = torch.load('checkpoint_all/best_fc2_weight.pt')

best_residual_bias = torch.load('checkpoint_all/best_residual_bias.pt')
best_residual_weight = torch.load('checkpoint_all/best_residual_weight.pt')

best_lstm_weight_ih0 = torch.load('checkpoint_all/best_lstm_weight_ih0.pt')
best_lstm_weight_hh0 = torch.load('checkpoint_all/best_lstm_weight_hh0.pt')
best_lstm_bias_ih0 = torch.load('checkpoint_all/best_lstm_bias_ih0.pt')
best_lstm_bias_hh0 = torch.load('checkpoint_all/best_lstm_bias_hh0.pt')

best_lstm_weight_ih1 = torch.load('checkpoint_all/best_lstm_weight_ih1.pt')
best_lstm_weight_hh1 = torch.load('checkpoint_all/best_lstm_weight_hh1.pt')
best_lstm_bias_ih1 = torch.load('checkpoint_all/best_lstm_bias_ih1.pt')
best_lstm_bias_hh1 = torch.load('checkpoint_all/best_lstm_bias_hh1.pt')

best_lstm_weight_hh2 = torch.load('checkpoint_all/best_lstm_weight_hh2.pt')
best_lstm_weight_ih2 = torch.load('checkpoint_all/best_lstm_weight_ih2.pt')
best_lstm_bias_ih2 = torch.load('checkpoint_all/best_lstm_bias_ih2.pt')
best_lstm_bias_hh2 = torch.load('checkpoint_all/best_lstm_bias_hh2.pt')

best_lstm_weight_hh3 = torch.load('checkpoint_all/best_lstm_weight_hh3.pt')
best_lstm_weight_ih3 = torch.load('checkpoint_all/best_lstm_weight_ih3.pt')
best_lstm_bias_ih3 = torch.load('checkpoint_all/best_lstm_bias_ih3.pt')
best_lstm_bias_hh3 = torch.load('checkpoint_all/best_lstm_bias_hh3.pt')

# columns
dataframe_headers = list(read_dataframe('all').columns)[1:]

print("### LSTM section")

print("weight abs sum of input gate across layers for above abg model")
avg_ih0_input_weight_abs_sum = best_lstm_weight_ih0[indices][:, 0:64, :].abs().mean(dim=1).mean(dim=0).tolist()
avg_ih1_input_weight_abs_sum = best_lstm_weight_ih1[indices][:, 0:64, :].abs().mean(dim=1).mean(dim=0).tolist()
avg_ih2_input_weight_abs_sum = best_lstm_weight_ih2[indices][:, 0:64, :].abs().mean(dim=1).mean(dim=0).tolist()
avg_ih3_input_weight_abs_sum = best_lstm_weight_ih3[indices][:, 0:64, :].abs().mean(dim=1).mean(dim=0).tolist()
print(sum(avg_ih0_input_weight_abs_sum))
print(sum(avg_ih1_input_weight_abs_sum))
print(sum(avg_ih2_input_weight_abs_sum))
print(sum(avg_ih3_input_weight_abs_sum))
print("in general, latter layer input gate has higher weights")
print()

print("weight abs sum of forget gate across layers for above abg model")
avg_ih0_forget_weight_abs_sum = best_lstm_weight_ih0[indices][:, 64:64*2, :].abs().mean(dim=1).mean(dim=0).tolist()
avg_ih1_forget_weight_abs_sum = best_lstm_weight_ih1[indices][:, 64:64*2, :].abs().mean(dim=1).mean(dim=0).tolist()
avg_ih2_forget_weight_abs_sum = best_lstm_weight_ih2[indices][:, 64:64*2, :].abs().mean(dim=1).mean(dim=0).tolist()
avg_ih3_forget_weight_abs_sum = best_lstm_weight_ih3[indices][:, 64:64*2, :].abs().mean(dim=1).mean(dim=0).tolist()
print(sum(avg_ih0_forget_weight_abs_sum))
print(sum(avg_ih1_forget_weight_abs_sum))
print(sum(avg_ih2_forget_weight_abs_sum))
print(sum(avg_ih3_forget_weight_abs_sum))
print("the 2nd and 4th layer forget gate has higher weights, suggesting the first layer is not too important")
print()

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
print("for low training loss model, the residual layer has is slightly helpful to the output")
print()

