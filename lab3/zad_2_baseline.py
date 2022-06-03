from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from lab3.lab_utils import load_dataset, VECTOR_PATH, evaluate, load_data_loaders, train
from lab3.models import BaselineModel


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


args = Args(300,
            10,
            32,
            32,
            2022,
            10,
            1e-4,
            0.25)

if __name__ == '__main__':
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    train_dataset, valid_dataset, test_dataset = load_dataset()
    train_dataloader, valid_dataloader, test_dataloader = load_data_loaders(train_dataset,
                                                                            valid_dataset,
                                                                            test_dataset,
                                                                            args)

    embedding_matrix = train_dataset.get_embedding_matrix(VECTOR_PATH)

    model = BaselineModel(embedding_matrix, args.embedding_size)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        print(f'Epoch {epoch + 1}.')
        train(model, train_dataloader, optimizer, criterion, max_norm=args.max_norm)
        print(f'Evaluating on validation dataset:')
        evaluate(model, valid_dataloader, criterion)

    print(f'Evaluating on test dataset:')
    evaluate(model, test_dataloader, criterion)
