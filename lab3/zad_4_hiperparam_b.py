from dataclasses import dataclass
from itertools import product

import numpy as np
import torch
from torch import nn

from lab3.lab_utils import load_dataset, evaluate, load_data_loaders, train
from lab3.models import RnnModel


@dataclass
class Args:
    embedding_sizes: [int]
    train_batch_size: int
    valid_batch_size: int
    test_batch_size: int
    seed: int
    epochs: int
    lrs: [float]
    max_norm: float
    rnn_hidden_sizes: list[int]
    rnn_num_layers: list[int]
    rnn_dropout: float
    rnn_bidirectional: bool
    rnn_types: [str]
    freezes: [bool]


args = Args([50, 300, 1000],
            10,
            32,
            32,
            2022,
            5,
            [1e-3, 5e-5, 1e-6],
            0.25,
            [500, 200, 20],
            [2, 4, 10],
            0.1,
            True,
            ["plain", "gru", "lstm"],
            [True, False])

if __name__ == '__main__':
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    train_dataset, valid_dataset, test_dataset = load_dataset(vocab_min_freq=1, vocab_max_size=-1)
    train_dataloader, valid_dataloader, test_dataloader = load_data_loaders(train_dataset,
                                                                            valid_dataset,
                                                                            test_dataset,
                                                                            args)

    with open("zad4_b.csv", "w") as file:
        print("Accuracy,F1 Score,Avg Loss,Rnn Type,Embedding Size,Learning Rate,Hidden Size,Num Layers,Freeze",
              file=file)

        for embedding_size, lr, hidden_size, num_layers, freeze, rnn_type in product(args.embedding_sizes,
                                                                                     args.lrs,
                                                                                     args.rnn_hidden_sizes,
                                                                                     args.rnn_num_layers,
                                                                                     args.freezes,
                                                                                     args.rnn_types):
            if hidden_size > embedding_size:
                continue
            print(f"{rnn_type},{embedding_size},{lr},{hidden_size},{num_layers},{freeze}")

            embedding_matrix = train_dataset.get_embedding_matrix(vector_size=embedding_size, freeze=freeze)

            model = RnnModel(embedding_matrix, embedding_size, hidden_size,
                             num_layers, args.rnn_dropout, args.rnn_bidirectional,
                             rnn_type)

            criterion = nn.BCEWithLogitsLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)

            for epoch in range(args.epochs):
                print(f'Epoch {epoch + 1}.')
                train(model, train_dataloader, optimizer, criterion, max_norm=args.max_norm, print_progress=False)
                print(f'Evaluating on validation dataset:')
                evaluate(model, valid_dataloader, criterion)

            print(f'Evaluating on test dataset:')
            results = evaluate(model, test_dataloader, criterion)
            accuracy = results["accuracy"]
            f1_score = results["f1_score"]
            avg_loss = results["avg_loss"]

            print(
                f"{accuracy:.2f},{f1_score:.2f},{avg_loss:.2f},{rnn_type},{embedding_size},"
                f"{lr},{hidden_size},{num_layers},{freeze}", file=file)
            file.flush()
