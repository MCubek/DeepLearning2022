from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from lab3 import dataset
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

    train_dataset = dataset.NLPDataset.from_file('data/sst_train_raw.csv')
    text_vocab = train_dataset.text_vocab
    labels_vocab = train_dataset.labels_vocab
    test_dataset = dataset.NLPDataset.from_file('data/sst_test_raw.csv', text_vocab, labels_vocab)
    val_dataset = dataset.NLPDataset.from_file('data/sst_valid_raw.csv', text_vocab, labels_vocab)
    train_dataloader, valid_dataloader, test_dataloader = load_data_loaders(train_dataset,
                                                                            val_dataset,
                                                                            test_dataset,
                                                                            args)
    text_vocab = train_dataset.text_vocab
    embedding_matrix = text_vocab.create_embedding_matrix(args.embedding_size, path_to_embeddings='data/sst_glove_6b_300d.txt')

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
