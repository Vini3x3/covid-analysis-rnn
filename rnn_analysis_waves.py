import copy
import os

import numpy as np
import torch

from lib.uilt_module import CurveLoss
from loader.DataLoader import read_dataframe
from loader.DataTransformer import lag_list
from model.Attention_LstmModel import Attention_ResidualLstmModel
from model.ScalingResidualLstmModel import ScalingResidualLstmModel

# script parameter
LAG = 15
WAVE = 4
REPEAT = 100

# prepare data
sequence = read_dataframe('all').to_numpy()
# sequence = read_dataframe('count').to_numpy()
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

# build model
input_dim = x_train.shape[-1]
hidden_dim = 64
num_layers = 4
output_dim = 1

### looping for repetitive training
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
    optimiser = torch.optim.Adam(model.parameters(), lr=0.01)
    model.train()
    for epoch in range(1, num_epochs + 1):
        y_pred = model(x_train)
        loss = loss_fn(y_pred, y_train)
        if best_model_train_loss > loss.item():
            best_model_state = copy.deepcopy(model.state_dict())
            best_model_train_loss = loss.item()
        if epoch % 100 == 0:
            print("Epoch: %d | MSE: %.2E | RRSE: %.2E" % (epoch, loss.item(), np.sqrt(loss.item() / y_var)))
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

    # test
    model.load_state_dict(best_model_state)

    best_training_loss.append(best_model_train_loss)

    best_fc2.append(model.fc2)
    best_residual.append(model.residual)

    best_lstm_weight_ih0.append(model.lstm.weight_ih_l0.data)
    best_lstm_weight_hh0.append(model.lstm.weight_hh_l0.data)
    best_lstm_bias_ih0.append(model.lstm.bias_ih_l0.data)
    best_lstm_bias_hh0.append(model.lstm.bias_hh_l0.data)

    best_lstm_lag.append(model.scaling.lag_importance.data)

### save files
CHECKPOINT_DIR = 'checkpoint_wave' + str(WAVE)
if not os.path.exists(CHECKPOINT_DIR):
    os.mkdir(CHECKPOINT_DIR)

torch.save(torch.stack([_.bias.data for _ in best_fc2], dim=0), CHECKPOINT_DIR + "/best_fc2_bias.pt")
torch.save(torch.stack([_.weight.data for _ in best_fc2], dim=0), CHECKPOINT_DIR + "/best_fc2_weight.pt")

torch.save(torch.stack([_.bias.data for _ in best_residual], dim=0), CHECKPOINT_DIR + "/best_residual_bias.pt")
torch.save(torch.stack([_.weight.data for _ in best_residual], dim=0), CHECKPOINT_DIR + "/best_residual_weight.pt")

torch.save(torch.stack(best_lstm_weight_ih0, dim=0), CHECKPOINT_DIR + "/best_lstm_weight_ih0.pt")
torch.save(torch.stack(best_lstm_weight_hh0, dim=0), CHECKPOINT_DIR + "/best_lstm_weight_hh0.pt")
torch.save(torch.stack(best_lstm_bias_ih0, dim=0), CHECKPOINT_DIR + "/best_lstm_bias_ih0.pt")
torch.save(torch.stack(best_lstm_bias_hh0, dim=0), CHECKPOINT_DIR + "/best_lstm_bias_hh0.pt")

torch.save(torch.stack(best_lstm_lag, dim=0), CHECKPOINT_DIR + "/best_lstm_lag.pt")

np.save(CHECKPOINT_DIR + '/best_train_loss.npy', np.array(best_training_loss))
