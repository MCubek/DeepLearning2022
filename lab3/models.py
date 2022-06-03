import torch
from torch import nn


class BaselineModel(nn.Module):
    def __init__(self, embedding_matrix, embedding_size=300):
        super().__init__()
        self.embedding_matrix = embedding_matrix

        self.fc_1 = nn.Linear(embedding_size, 150)
        self.fc_2 = nn.Linear(150, 150)
        self.fc_3 = nn.Linear(150, 1)

    def forward(self, x):
        h = self.embedding_matrix(x)
        h = h.mean(dim=1)
        h = self.fc_1(h)
        h = torch.relu(h)
        h = self.fc_2(h)
        h = torch.relu(h)
        h = self.fc_3(h)
        return h.flatten()
