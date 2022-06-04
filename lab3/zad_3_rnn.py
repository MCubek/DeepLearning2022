from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from lab3.lab_utils import load_dataset, VECTOR_PATH, evaluate, load_data_loaders, train
from lab3.models import BaselineModel, LstmRnnModel


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
    rnn_hidden_size: int
    rnn_num_layers: int
    rnn_dropout: float
    rnn_bidirectional: bool


args = Args(300,
            10,
            32,
            32,
            2022,
            100,
            1e-5,
            0.25,
            150,
            2,
            0,
            False)

if __name__ == '__main__':
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    train_dataset, valid_dataset, test_dataset = load_dataset(vocab_min_freq=1, vocab_max_size=-1)
    train_dataloader, valid_dataloader, test_dataloader = load_data_loaders(train_dataset,
                                                                            valid_dataset,
                                                                            test_dataset,
                                                                            args)

    embedding_matrix = train_dataset.get_embedding_matrix(VECTOR_PATH)

    model = LstmRnnModel(embedding_matrix, args)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        print(f'Epoch {epoch + 1}.')
        train(model, train_dataloader, optimizer, criterion, max_norm=args.max_norm)
        print(f'Evaluating on validation dataset:')
        evaluate(model, valid_dataloader, criterion)

    print(f'Evaluating on test dataset:')
    evaluate(model, test_dataloader, criterion)
