import copy

import numpy as np
import torch

from lib.uilt_module import CurveLoss
from loader.DataLoader import read_dataframe
from loader.DataTransformer import lag_list
from model.ResidualLstmModel import ResidualLstmModel

# script parameter
LAG = 16

# prepare data
sequence = read_dataframe('all').to_numpy()
# sequence = read_dataframe('count').to_numpy()
y_var = np.var(sequence[:, -1])
shifted_sequence = lag_list(sequence, LAG)  # shift into delayed sequences

x_train = shifted_sequence[:, :-1, 1:]  # for each delayed sequence, take all elements except last element
y_train = shifted_sequence[:, -1, -1]  # for each delayed sequence, only take the last element
y_train = y_train.reshape(-1, 1)

x_train = torch.from_numpy(x_train.astype('float64')).type(torch.Tensor)  # convert to tensor
y_train = torch.from_numpy(y_train.astype('int32')).type(torch.Tensor)  # convert to tensor

# build model
input_dim = x_train.shape[-1]
hidden_dim = 64
num_layers = 4
output_dim = 1

### looping for repetitive training
# best_models = []
best_training_loss = []

best_fc1 = []
best_residual = []
best_fc2 = []

best_lstm_weight_ih0 = []
best_lstm_weight_hh0 = []
best_lstm_bias_ih0 = []
best_lstm_bias_hh0 = []

best_lstm_weight_ih1 = []
best_lstm_weight_hh1 = []
best_lstm_bias_ih1 = []
best_lstm_bias_hh1 = []

best_lstm_weight_ih2 = []
best_lstm_weight_hh2 = []
best_lstm_bias_ih2 = []
best_lstm_bias_hh2 = []

best_lstm_weight_ih3 = []
best_lstm_weight_hh3 = []
best_lstm_bias_ih3 = []
best_lstm_bias_hh3 = []

REPEAT = 100

for _ in range(REPEAT):
    best_model_state = None
    best_model_train_loss = np.inf
    model = ResidualLstmModel(input_dim, hidden_dim, num_layers, output_dim)

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

    # best_models.append(best_model_state)
    best_training_loss.append(best_model_train_loss)

    best_fc1.append(model.fc)
    best_fc2.append(model.fc2)
    best_residual.append(model.residual)

    best_lstm_weight_ih0.append(model.lstm.weight_ih_l0.data)
    best_lstm_weight_hh0.append(model.lstm.weight_hh_l0.data)
    best_lstm_bias_ih0.append(model.lstm.bias_ih_l0.data)
    best_lstm_bias_hh0.append(model.lstm.bias_hh_l0.data)

    best_lstm_weight_ih1.append(model.lstm.weight_ih_l1.data)
    best_lstm_weight_hh1.append(model.lstm.weight_hh_l1.data)
    best_lstm_bias_ih1.append(model.lstm.bias_ih_l1.data)
    best_lstm_bias_hh1.append(model.lstm.bias_hh_l1.data)

    best_lstm_weight_ih2.append(model.lstm.weight_ih_l2.data)
    best_lstm_weight_hh2.append(model.lstm.weight_hh_l2.data)
    best_lstm_bias_ih2.append(model.lstm.bias_ih_l2.data)
    best_lstm_bias_hh2.append(model.lstm.bias_hh_l2.data)

    best_lstm_weight_ih3.append(model.lstm.weight_ih_l3.data)
    best_lstm_weight_hh3.append(model.lstm.weight_hh_l3.data)
    best_lstm_bias_ih3.append(model.lstm.bias_ih_l3.data)
    best_lstm_bias_hh3.append(model.lstm.bias_hh_l3.data)

### save files
torch.save(torch.stack([_.bias.data for _ in best_fc1], dim=0), "checkpoint_all/best_fc1_bias.pt")
torch.save(torch.stack([_.weight.data for _ in best_fc1], dim=0), "checkpoint_all/best_fc1_weight.pt")

torch.save(torch.stack([_.bias.data for _ in best_fc2], dim=0), "checkpoint_all/best_fc2_bias.pt")
torch.save(torch.stack([_.weight.data for _ in best_fc2], dim=0), "checkpoint_all/best_fc2_weight.pt")

torch.save(torch.stack([_.bias.data for _ in best_residual], dim=0), "checkpoint_all/best_residual_bias.pt")
torch.save(torch.stack([_.weight.data for _ in best_residual], dim=0), "checkpoint_all/best_residual_weight.pt")

torch.save(torch.stack(best_lstm_weight_ih0, dim=0), "checkpoint_all/best_lstm_weight_ih0.pt")
torch.save(torch.stack(best_lstm_weight_hh0, dim=0), "checkpoint_all/best_lstm_weight_hh0.pt")
torch.save(torch.stack(best_lstm_bias_ih0, dim=0), "checkpoint_all/best_lstm_bias_ih0.pt")
torch.save(torch.stack(best_lstm_bias_hh0, dim=0), "checkpoint_all/best_lstm_bias_hh0.pt")

torch.save(torch.stack(best_lstm_weight_ih1, dim=0), "checkpoint_all/best_lstm_weight_ih1.pt")
torch.save(torch.stack(best_lstm_weight_hh1, dim=0), "checkpoint_all/best_lstm_weight_hh1.pt")
torch.save(torch.stack(best_lstm_bias_ih1, dim=0), "checkpoint_all/best_lstm_bias_ih1.pt")
torch.save(torch.stack(best_lstm_bias_hh1, dim=0), "checkpoint_all/best_lstm_bias_hh1.pt")

torch.save(torch.stack(best_lstm_weight_ih2, dim=0), "checkpoint_all/best_lstm_weight_ih2.pt")
torch.save(torch.stack(best_lstm_weight_hh2, dim=0), "checkpoint_all/best_lstm_weight_hh2.pt")
torch.save(torch.stack(best_lstm_bias_ih2, dim=0), "checkpoint_all/best_lstm_bias_ih2.pt")
torch.save(torch.stack(best_lstm_bias_hh2, dim=0), "checkpoint_all/best_lstm_bias_hh2.pt")

torch.save(torch.stack(best_lstm_weight_ih3, dim=0), "checkpoint_all/best_lstm_weight_ih3.pt")
torch.save(torch.stack(best_lstm_weight_hh3, dim=0), "checkpoint_all/best_lstm_weight_hh3.pt")
torch.save(torch.stack(best_lstm_bias_ih3, dim=0), "checkpoint_all/best_lstm_bias_ih3.pt")
torch.save(torch.stack(best_lstm_bias_hh3, dim=0), "checkpoint_all/best_lstm_bias_hh3.pt")

np.save('checkpoint_all/best_train_loss.npy', np.array(best_training_loss))
