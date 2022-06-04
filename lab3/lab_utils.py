import numpy as np
import torch
from torch.utils.data import DataLoader

from lab3 import dataset

PRINT_LOSS_N = 100

TRAIN_PATH = 'data/sst_train_raw.csv'
VALID_PATH = 'data/sst_valid_raw.csv'
TEST_PATH = 'data/sst_test_raw.csv'
VECTOR_PATH = 'data/sst_glove_6b_300d.txt'


def load_dataset(vocab_min_freq=0, vocab_max_size=-1):
    train_dataset = dataset.NLPDataset.from_file(TRAIN_PATH, vocab_max_size=vocab_max_size,
                                                 vocab_min_freq=vocab_min_freq)

    text_vocab = train_dataset.text_vocab
    label_vocab = train_dataset.label_vocab

    test_dataset = dataset.NLPDataset.from_file(TEST_PATH, text_vocab, label_vocab)
    valid_dataset = dataset.NLPDataset.from_file(VALID_PATH, text_vocab, label_vocab)

    return train_dataset, valid_dataset, test_dataset


def load_data_loaders(train_dataset, valid_dataset, test_dataset, args):
    train_dataloader = DataLoader(train_dataset, args.train_batch_size, shuffle=True, collate_fn=dataset.pad_collate_fn)
    valid_dataloader = DataLoader(valid_dataset, args.valid_batch_size, shuffle=True, collate_fn=dataset.pad_collate_fn)
    test_dataloader = DataLoader(test_dataset, args.test_batch_size, shuffle=True, collate_fn=dataset.pad_collate_fn)

    return train_dataloader, valid_dataloader, test_dataloader


def evaluate(model, data, criterion):
    model.eval()
    confusion_matrix = np.zeros((2, 2))
    losses = []

    with torch.no_grad():
        for batch_num, (data, target, lengths) in enumerate(data):
            logits = model.forward(data, lengths)
            loss = criterion(logits, target)
            losses.append(loss.item())

            predicted = torch.sigmoid(logits).round().int().numpy()
            target_np = target.int().numpy()

            for i in range(len(data)):
                confusion_matrix[target_np[i].item()][predicted[i].item()] += 1

    accuracy = (confusion_matrix[0][0] + confusion_matrix[1][1]) / np.sum(confusion_matrix)
    recall = confusion_matrix[1][1] / (confusion_matrix[1][0] + confusion_matrix[1][1])
    precision = confusion_matrix[1][1] / (confusion_matrix[0][1] + confusion_matrix[1][1])
    f1_score = 2 * precision * recall / (precision + recall)
    avg_loss = np.mean(losses).item()
    print(f"Accuracy: {accuracy:.2f}")
    print(f"F1 Score: {f1_score:.2f}")
    print(f"Avg Loss {avg_loss:.2f}")
    print(f"Confusion Matrix:\n{confusion_matrix}")

    return {
        "accuracy": accuracy,
        "f1_score": f1_score,
        "avg_loss": avg_loss,
        "confusion_matrix": confusion_matrix
    }


def train(model, data, optimizer, criterion, max_norm, print_progress=True):
    model.train()
    for batch_num, (data, target, lengths) in enumerate(data):
        model.zero_grad()
        logits = model.forward(data, lengths)
        loss = criterion(logits, target)
        loss.backward()
        if max_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()
        if print_progress and batch_num % PRINT_LOSS_N == 0:
            print(f"Iter: {batch_num}, Loss: {loss.item():.3f}")
