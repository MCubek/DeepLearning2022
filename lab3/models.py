import torch
from torch import nn


class BaselineModel(nn.Module):
    def __init__(self, embedding_matrix, embedding_size=300):
        super().__init__()
        self.embedding_matrix = embedding_matrix

        self.fc_1 = nn.Linear(embedding_size, 150)
        self.fc_2 = nn.Linear(150, 150)
        self.fc_3 = nn.Linear(150, 1)

    def forward(self, x, length):
        h = self.embedding_matrix(x)
        h = h.mean(dim=1)
        h = self.fc_1(h)
        h = torch.relu(h)
        h = self.fc_2(h)
        h = torch.relu(h)
        h = self.fc_3(h)
        return h.flatten()


class LstmRnnModel(nn.Module):
    def __init__(self, embedding_matrix, args):
        super().__init__()
        self.embedding_matrix = embedding_matrix

        self.lstm = nn.LSTM(input_size=args.embedding_size,
                            hidden_size=args.rnn_hidden_size,
                            num_layers=args.rnn_num_layers,
                            dropout=args.rnn_dropout,
                            bidirectional=args.rnn_bidirectional)

        self.fc_1 = nn.Linear(args.rnn_hidden_size, args.rnn_hidden_size)
        self.fc_2 = nn.Linear(args.rnn_hidden_size, 1)

    def forward(self, x, length):
        h = self.embedding_matrix(x)

        _, (h_lstm, _) = self.lstm(h)
        h = h_lstm[-1].reshape((h_lstm.shape[1], h_lstm.shape[2]))

        h = self.fc_1(h)
        h = torch.relu(h)
        h = self.fc_2(h)
        return h.flatten()
