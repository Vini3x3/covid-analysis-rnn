import numpy as np
import torch

from loader.DataLoader import read_dataframe
from loader.DataTransformer import lag_list, transform_matrix
from model.CnnLstmModel import CnnLstmModel
from model.FcLstmModel import FcLstmModel
from model.LstmModel import LstmModel

# script parameter
# MODE: MA (moving average), D1(lag 1 degree), DMA(decaying moving average) or default no change
MODE = 'NORM'
LAG = 16

# prepare data
sequence = read_dataframe('all').to_numpy()
sequence = sequence[:, 1:]
sequence = transform_matrix(sequence, 'DIFF')
sequence = transform_matrix(sequence, 'NORM')
sequence[np.isnan(sequence)] = 0 # fill na - there is a column which are all 0

y_var = np.var(sequence[:,-1])
shifted_sequence = lag_list(sequence, LAG)  # shift into delayed sequences

x_train = shifted_sequence[:, :-1, 1:]  # for each delayed sequence, take all elements except last element
y_train = shifted_sequence[:, -1, -1]  # for each delayed sequence, only take the last element
y_train = y_train.reshape(-1, 1)

x_train = torch.from_numpy(x_train.astype('float64')).type(torch.Tensor)  # convert to tensor
y_train = torch.from_numpy(y_train.astype('float64')).type(torch.Tensor)  # convert to tensor

# build model
input_dim = x_train.shape[-1]
hidden_dim = 64
num_layers = 2
output_dim = 1

# model = CnnLstmModel(input_dim, LAG)
# model = FcLstmModel(input_dim, hidden_dim, num_layers, output_dim, LAG - 1, 0, 0)
model = LstmModel(input_dim, hidden_dim, num_layers, output_dim)

# train
num_epochs = 3_000
loss_fn = torch.nn.MSELoss()
optimiser = torch.optim.Adam(model.parameters(), lr=0.001)
model.train()
for epoch in range(1, num_epochs + 1):
    y_pred = model(x_train)
    loss = loss_fn(y_pred, y_train)
    if epoch % 100 == 0:
        print("Epoch: %d | MSE: %.2E | RRSE: %.2E" % (epoch, loss.item(), np.sqrt(loss.item() / y_var)))
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
