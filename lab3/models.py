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


class RnnModel(nn.Module):
    def lstm_forward(self, x):
        _, (h, _) = self.rnn(x)

        return h[-1]

    def plain_rnn_forward(self, x):
        _, h = self.rnn(x)

        return h[-1]

    def __init__(self, embedding_matrix, args):
        super().__init__()
        self.embedding_matrix = embedding_matrix

        rnn_params = {
            "input_size": args.embedding_size,
            "hidden_size": args.rnn_hidden_size,
            "num_layers": args.rnn_num_layers,
            "dropout": args.rnn_dropout,
            "bidirectional": args.rnn_bidirectional
        }

        if args.rnn_type == "lstm":
            self.rnn = nn.LSTM(**rnn_params)
            self.rnn_forward = self.lstm_forward
        elif args.rnn_type == "plain":
            self.rnn = nn.RNN(**rnn_params)
            self.rnn_forward = self.plain_rnn_forward
        elif args.rnn_type == "gru":
            self.rnn = nn.GRU(**rnn_params)
            self.rnn_forward = self.plain_rnn_forward
        else:
            raise AttributeError

        self.fc_1 = nn.Linear(args.rnn_hidden_size, args.rnn_hidden_size)
        self.fc_2 = nn.Linear(args.rnn_hidden_size, 1)

    def forward(self, x, length):
        batch = self.embedding_matrix(x)
        batch_t = torch.transpose(batch, 0, 1)
        h = self.rnn_forward(batch_t)
        h = self.fc_1(h)
        s = torch.relu(h)
        h = self.fc_2(s)
        return h.flatten()
