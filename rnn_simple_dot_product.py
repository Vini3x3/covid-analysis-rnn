import numpy as np
import torch

from model.LstmModel import LstmModel

"""
Scenario: 
To train a RNN model that can calculate dot product between two array, 
e.g. given [3,4,5] and [6,7,8], it can predict 3*6+4*7+5*8=86 
"""

# prepare data

num_samples = 200
seq_length = 5

x1 = np.random.uniform(0, 1, (num_samples, seq_length))
x2 = np.random.uniform(0, 1, (num_samples, seq_length))

y_train = np.sum(x1 * x2, axis=1).reshape(-1, 1)
X_train = np.stack((x1, x2), axis=2)

X_train = torch.from_numpy(X_train).type(torch.Tensor)  # convert to tensor
y_train = torch.from_numpy(y_train).type(torch.Tensor)  # convert to tensor

# build model
input_dim = 2
hidden_dim = 64
num_layers = 3
output_dim = 1

model = LstmModel(input_dim, hidden_dim, num_layers, output_dim)

# train
num_epochs = 2_000
loss_fn = torch.nn.MSELoss()
optimiser = torch.optim.Adam(model.parameters(), lr=0.005)
model.train()
for epoch in range(1, num_epochs + 1):
    y_pred = model(X_train)
    loss = loss_fn(y_pred, y_train)
    if epoch % 100 == 0:
        print("Epoch: %d | MSE: %.2E" % (epoch, loss.item()))
    optimiser.zero_grad()
    loss.backward()
    optimiser.step()

# test
model.eval()
y_pred = model(X_train[:5])

y_pred = y_pred.detach().numpy()  # revert from tensor
y_pred = y_pred.reshape(-1)  # reshape back to normal list
print("sample prediction:  ", y_pred[:5])

y_train_sample = y_train[:5].detach().numpy().reshape(-1)
print("sample true result: ", y_train_sample)

y_pred_round = [round(p, 1) for p in y_pred]
y_train_round = [round(p, 1) for p in y_train_sample]
assert (y_pred_round == y_train_round)

# verify
sample_x1 = sample_x2 = [0.1, 0.3, 0.4, 0.5, 0.7]
sample_answer = 1

sample_input = np.stack((x1, x2), axis=2)
sample_input = torch.from_numpy(sample_input).type(torch.Tensor)

sample_pred = model(sample_input)

sample_pred = sample_pred.detach().numpy()  # revert from tensor
sample_pred = sample_pred.reshape(-1)  # reshape back to normal list
sample_pred = sample_pred[0]  # extract only element

print("verify input: ", sample_x1, sample_x2)
print("verify prediction:  ", sample_pred)
print("verify true result: ", sample_answer)
assert (round(sample_pred) == sample_answer)
