from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from lab3.lab_utils import load_dataset, VECTOR_PATH, evaluate, load_data_loaders, train
from lab3.models import BaselineModel, RnnModel


@dataclass
class Args:
    embedding_size: int
    train_batch_size: int
    valid_batch_size: int
    test_batch_size: int
    seed: list[int]
    epochs: int
    lr: float
    max_norm: float
    rnn_hidden_size: int
    rnn_num_layers: int
    rnn_dropout: float
    rnn_bidirectional: bool
    rnn_type: str


args = Args(300,
            10,
            32,
            32,
            [2022, 1999, 2023, 9999, 1111],
            10,
            5e-5,
            0.25,
            150,
            2,
            0,
            False,
            "lstm")


def save_results_to_file(results: dict, file_path):
    result_string = ""
    for seed, result in results.items():
        result_string += f"""
Seed = {seed}
Accuracy = {result["accuracy"]:.2f}
F1 Score = {result["f1_score"]:.2f}
Avg Loss = {result["avg_loss"]:.2f}
Confusion Matrix =
{result["confusion_matrix"]}
        """

    with open(file_path, "w") as file:
        file.write(result_string)
        file.flush()


if __name__ == '__main__':

    results = {}

    for seed in args.seed:
        np.random.seed(seed)
        torch.manual_seed(seed)

        print(f'\nseed={seed}')

        train_dataset, valid_dataset, test_dataset = load_dataset(vocab_min_freq=1, vocab_max_size=-1)
        train_dataloader, valid_dataloader, test_dataloader = load_data_loaders(train_dataset,
                                                                                valid_dataset,
                                                                                test_dataset,
                                                                                args)

        embedding_matrix = train_dataset.get_embedding_matrix(VECTOR_PATH)

        model = RnnModel(embedding_matrix, args)

        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

        for epoch in range(args.epochs):
            print(f'Epoch {epoch + 1}.')
            train(model, train_dataloader, optimizer, criterion, max_norm=args.max_norm, print_progress=False)
            print(f'Evaluating on validation dataset:')
            evaluate(model, valid_dataloader, criterion)

        print(f'Evaluating on test dataset:')

        results[seed] = evaluate(model, test_dataloader, criterion)

    save_results_to_file(results, "./zad3.txt")
