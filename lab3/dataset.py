from collections import Counter
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset, DataLoader


@dataclass
class Instance:
    tokens: []
    target_class: str


class Vocab:
    PADDING = "<PAD>", 0
    UNKNOWN = "<UNK>", 1
    special = [PADDING[0], UNKNOWN[0]]

    def __init__(self, frequencies: dict, maxsize=-1, min_freq=0, is_label=False):
        self.stoi = {}
        self.itos = {}

        special_counter = 0
        if not is_label:
            for item in Vocab.special:
                self.stoi[item] = special_counter
                self.itos[special_counter] = item
                special_counter += 1

        for i, (item, count) in enumerate(sorted(frequencies.items(), key=lambda x: x[1], reverse=True)):
            if maxsize != -1 and i > maxsize:
                break
            if count < min_freq:
                continue

            self.stoi[item] = i + special_counter
            self.itos[i + special_counter] = item

    def encode_list(self, tokens):
        return torch.tensor([self.stoi.get(x, Vocab.UNKNOWN[1]) for x in tokens])

    def encode_item(self, token):
        return torch.tensor(self.stoi.get(token))

    def decode(self, indexes):
        return [self.itos.get(x, "<UNK>") for x in indexes]

    def generate_embedding_matrix(self, vector_size=300, matrix_path=None):
        vectors = {}

        if matrix_path:
            file = open(matrix_path, 'r')
            lines = file.readlines()

            for line in lines:
                split = line.split()
                token = split[0]
                vector_representations = split[1:]

                vectors[token] = [float(x) for x in vector_representations]

            file.close()

        embedding_matrix = torch.randn(len(self.stoi), vector_size)

        for word, index in self.stoi.items():
            if word not in vectors:
                continue

            if word == Vocab.PADDING[0]:
                embedding_matrix[index] = torch.zeros(vector_size)
            else:
                embedding_matrix[index] = torch.tensor(vectors[word])

        return torch.nn.Embedding.from_pretrained(embedding_matrix, freeze=True, padding_idx=0)


class NLPDataset(Dataset):

    def __init__(self, instances: list[Instance], text_vocab: Vocab, label_vocab: Vocab):
        self.text_vocab = text_vocab
        self.label_vocab = label_vocab

        self.instances = instances

    def __getitem__(self, index):
        instance = self.instances[index]
        return self.text_vocab.encode_list(instance.tokens), self.label_vocab.encode_item(instance.target_class)

    def __len__(self):
        return len(self.instances)

    def get_embedding_matrix(self, matrix_path, vector_size=300):
        return self.text_vocab.generate_embedding_matrix(vector_size, matrix_path)

    @classmethod
    def from_file(cls, data_path, token_vocab=None, label_vocab=None):
        instances = generate_instances_from_csv(data_path)

        if token_vocab is None or label_vocab is None:
            token_vocab, label_vocab = generate_vocab_from_instances(instances)

        return cls(instances, token_vocab, label_vocab)


def generate_instances_from_csv(csv_path) -> list[Instance]:
    instances = []

    with open(csv_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            split = line.split(",")
            words = [x.strip() for x in split[0].split()]
            target = split[1].strip()

            instances.append(Instance(words, target))

    return instances


def generate_vocab_from_instances(instances: list[Instance], maxsize=-1, min_freq=0) -> (Vocab, Vocab):
    token_frequencies = Counter()
    label_frequencies = Counter()

    for instance in instances:
        label_frequencies[instance.target_class] += 1

        for token in instance.tokens:
            token_frequencies[token] += 1

    token_vocab = Vocab(token_frequencies, maxsize, min_freq)
    label_vocab = Vocab(label_frequencies, maxsize, min_freq, is_label=True)

    return token_vocab, label_vocab


def pad_collate_fn(batch, pad_index=0):
    texts, labels = zip(*batch)  # Assuming the instance is in tuple-like form
    lengths = torch.tensor([len(text) for text in texts])  # Needed for later
    padded_texts = torch.nn.utils.rnn.pad_sequence(texts, batch_first=True, padding_value=pad_index)
    labels = torch.tensor(labels, dtype=torch.float32)
    return padded_texts, labels, lengths


if __name__ == '__main__':
    batch_size = 2  # Only for demonstrative purposes
    shuffle = False  # Only for demonstrative purposes
    train_dataset = NLPDataset.from_file('data/sst_train_raw.csv')
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size,
                                  shuffle=shuffle, collate_fn=pad_collate_fn)
    texts, labels, lengths = next(iter(train_dataloader))

    instance_text = train_dataset.instances[3].tokens
    instance_label = train_dataset.instances[3].target_class

    print(f"Text: {instance_text}")
    print(f"Label: {instance_label}")

    numericalized_text, numericalized_label = train_dataset[3]
    print(f"Numericalized text: {numericalized_text}")
    print(f"Numericalized label: {numericalized_label}")

    print(f"Texts: {texts}")
    print(f"Labels: {labels}")
    print(f"Lengths: {lengths}")
