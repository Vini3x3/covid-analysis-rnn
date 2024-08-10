import pandas as pd
import torch

from loader.DataTransformer import lag_list
from model.LstmModel import LstmModel


# prepare data
def get_date_count(df: pd.DataFrame, col: str) -> pd.DataFrame:
    agg_df = df.groupby(col)[col].count()
    date_idx = pd.date_range(agg_df.index.min(), agg_df.index.max())
    agg_series = pd.Series(agg_df)
    agg_series.index = pd.DatetimeIndex(agg_series.index)
    agg_series = agg_series.reindex(date_idx, fill_value=0)
    return pd.DataFrame({col: agg_series.index, 'count': agg_series.values})


df_infected = pd.read_csv("data/covid/covid_hk_case_std.csv")
df_infected['report_date'] = pd.to_datetime(df_infected['report_date'], format='%Y%m%d')  # convert to datetime type
print(df_infected.shape)

df_count = get_date_count(df_infected, 'report_date')
print(df_count.head())
print(df_count.shape)

sequence = df_count['count'].to_numpy().reshape(-1, 1)
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

model = LstmModel(input_dim, hidden_dim, num_layers, output_dim)

# train
num_epochs = 3_000
loss_fn = torch.nn.MSELoss()
optimiser = torch.optim.Adam(model.parameters(), lr=0.01)
model.train()
for epoch in range(1, num_epochs + 1):
    y_pred = model(x_train)
    loss = loss_fn(y_pred, y_train)
    if epoch % 100 == 0:
        print("Epoch: %d | MSE: %.2E" % (epoch, loss.item()))
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
