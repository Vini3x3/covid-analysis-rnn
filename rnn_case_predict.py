import copy

import numpy as np
import torch

from loader.DataLoader import read_sequence
from loader.DataTransformer import lag_list
from model.FcLstmModel import FcLstmModel

# prepare data
sequence = read_sequence('case')
y_var = np.var(sequence)
sequence = sequence.reshape(-1, 1)
shifted_sequence = lag_list(sequence, 16)  # shift into delayed sequences

x_train = shifted_sequence[:, :-1, :]  # for each delayed sequence, take all elements except last element
y_train = shifted_sequence[:, -1, :]  # for each delayed sequence, only take the last element

x_train = torch.from_numpy(x_train).type(torch.Tensor)  # convert to tensor
y_train = torch.from_numpy(y_train).type(torch.Tensor)  # convert to tensor

# build model
input_dim = 1
hidden_dim = 64
num_layers = 2
output_dim = 1

model = FcLstmModel(input_dim, hidden_dim, num_layers, output_dim, 16 - 1, 0, 0)

# train
num_epochs = 2_000
loss_fn = torch.nn.MSELoss()
optimiser = torch.optim.Adam(model.parameters(), lr=0.01)
min_loss = np.inf
best_model_state = None
model.train()
for epoch in range(1, num_epochs + 1):
    y_pred = model(x_train)
    loss = loss_fn(y_pred, y_train)
    if epoch % 100 == 0:
        print("Epoch: %d | MSE: %.2E | RRSE: %.2E" % (epoch, loss.item(), np.sqrt(loss.item() / y_var)))
    if min_loss > loss.item():
        best_model_state = copy.deepcopy(model.state_dict())
        min_loss = loss.item()
        # save(model.state_dict(), 'rnn_case_predict.chk', ['min_loss=' + str(min_loss), 'epoch='+str(epoch)])
    optimiser.zero_grad()
    loss.backward()
    optimiser.step()

# predict
model.eval()
# load('rnn_case_predict.chk', model)
model.load_state_dict(best_model_state)
x_test = x_train[0].reshape(1, 15, 1)
y_pred = []
prediction_range = 30 # x_train.shape[0]
for _ in range(prediction_range):
    _ = model(x_test)
    x_test = torch.cat((x_test[0][1:], _), dim=0)
    x_test = x_test.reshape(1, 15, 1)
    _2 = _.detach().numpy()  # revert from tensor
    _2 = _2.reshape(-1)  # reshape back to normal list
    y_pred.append(_2)

y_pred = np.array(y_pred).reshape(-1)  # reshape back to normal list
print("sample prediction:  ", y_pred)

y_train_sample = y_train.detach().numpy().reshape(-1)[:prediction_range]
print("sample true result: ", y_train_sample)

mse = sum((y_train_sample - y_pred) ** 2) / len(y_pred)
print("prediction MSE: ", mse)
