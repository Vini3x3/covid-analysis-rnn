import numpy as np
import torch
from torch import nn


class RandomForestLstmModel(nn.Module):
    def __init__(self, models, performance = np.ones(0)):
        super(RandomForestLstmModel, self).__init__()
        self.models = models
        self.weightings = np.ones(len(models))
        if len(performance) == len(models):
            _ = sum(performance) / performance
            self.weightings = np.array([_2 / sum(_) for _2 in _])

    def forward(self, x):
        model_results = []
        for model in self.models:
            model_results.append(model(x))
        model_results = torch.stack([_ for _ in model_results], dim=0)
        weighted_model_results = [model_results[i] * self.weightings[i] for i in range(len(self.models))]
        return torch.stack(weighted_model_results, dim=0).mean(dim=0)
