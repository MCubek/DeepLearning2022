from dataclasses import dataclass
from itertools import product

import numpy as np
import torch
from torch import nn

from lab3.lab_utils import load_dataset, VECTOR_PATH, evaluate, load_data_loaders, train
from lab3.models import RnnModel


@dataclass
class Args:
    embedding_size: int
    train_batch_size: int
    valid_batch_size: int
    test_batch_size: int
    seed: int
    epochs: int
    lr: float
    max_norm: float
    rnn_hidden_sizes: list[int]
    rnn_num_layers: list[int]
    rnn_dropouts: list[float]
    rnn_bidirectional: list[bool]
    rnn_types: list[str]


args = Args(300,
            10,
            32,
            32,
            2022,
            10,
            5e-5,
            0.25,
            [300, 150, 10],
            [2, 4, 10],
            [0, 0.1, 0.25],
            [False, True],
            ["plain", "gru", "lstm"])

if __name__ == '__main__':
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    train_dataset, valid_dataset, test_dataset = load_dataset(vocab_min_freq=1, vocab_max_size=-1)
    train_dataloader, valid_dataloader, test_dataloader = load_data_loaders(train_dataset,
                                                                            valid_dataset,
                                                                            test_dataset,
                                                                            args)

    embedding_matrix = train_dataset.get_embedding_matrix(VECTOR_PATH)

    with open("zad4_a.csv", "w") as file:
        print("Accuracy,F1 Score,Avg Loss,RNN Type,Hidden Size,Num Layers,Dropout,Bidirectional", file=file)

        for hidden_size, num_layers, dropout, bidirectional, rnn_type in product(args.rnn_hidden_sizes,
                                                                                 args.rnn_num_layers,
                                                                                 args.rnn_dropouts,
                                                                                 args.rnn_bidirectional,
                                                                                 args.rnn_types):
            print(f"{rnn_type},{hidden_size},{num_layers},{dropout},{bidirectional}")

            model = RnnModel(embedding_matrix, args.embedding_size, hidden_size,
                             num_layers, dropout, bidirectional,
                             rnn_type)

            criterion = nn.BCEWithLogitsLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

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
                f"{accuracy:.2f},{f1_score:.2f},{avg_loss:.2f},{rnn_type},"
                f"{hidden_size},{num_layers},{dropout},{bidirectional}", file=file)
            file.flush()
