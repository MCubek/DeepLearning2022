import time

import torch.optim
from torch.utils.data import DataLoader

from dataset import MNISTMetricDataset
from model import SimpleMetricEmbedding, IdentityModel
from utils import train, evaluate, compute_representations, train_identity

EVAL_ON_TEST = True
EVAL_ON_TRAIN = False
MODEL_IDENTITY = False
WITHOUT_CLASS = True

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"= Using device {device}")

    # CHANGE ACCORDING TO YOUR PREFERENCE
    mnist_download_root = "./mnist/"
    ds_train = MNISTMetricDataset(mnist_download_root, split='train')
    ds_train_without_class = MNISTMetricDataset(mnist_download_root, split='train', remove_class=0)
    ds_test = MNISTMetricDataset(mnist_download_root, split='test')
    ds_traineval = MNISTMetricDataset(mnist_download_root, split='traineval')

    num_classes = 10

    print(f"> Loaded {len(ds_train)} training images!")
    print(f"> Loaded {len(ds_test)} validation images!")

    train_loader_full = DataLoader(
        ds_train,
        batch_size=64,
        shuffle=True,
        pin_memory=True,
        num_workers=4,
        drop_last=True
    )

    train_loader_without_class = DataLoader(
        ds_train_without_class,
        batch_size=64,
        shuffle=True,
        pin_memory=True,
        num_workers=4,
        drop_last=True
    )

    test_loader = DataLoader(
        ds_test,
        batch_size=1,
        shuffle=False,
        pin_memory=True,
        num_workers=1
    )

    traineval_loader = DataLoader(
        ds_traineval,
        batch_size=1,
        shuffle=False,
        pin_memory=True,
        num_workers=1
    )

    if MODEL_IDENTITY:
        emb_size = 28 * 28
        model = IdentityModel().to(device)
    else:
        emb_size = 32
        model = SimpleMetricEmbedding(1, emb_size).to(device)
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=1e-3
        )

    if WITHOUT_CLASS:
        train_loader = train_loader_without_class
    else:
        train_loader = train_loader_full

    epochs = 3
    for epoch in range(epochs):
        print(f"Epoch: {epoch}")
        t0 = time.time_ns()

        if MODEL_IDENTITY:
            train_loss = train_identity(model, train_loader, device)
        else:
            train_loss = train(model, optimizer, train_loader, device)

        print(f"Mean Loss in Epoch {epoch}: {train_loss:.3f}")
        if EVAL_ON_TEST or EVAL_ON_TRAIN:
            print("Computing mean representations for evaluation...")
            representations = compute_representations(model, train_loader_full, num_classes, emb_size, device)
        if EVAL_ON_TRAIN:
            print("Evaluating on training set...")
            acc1 = evaluate(model, representations, traineval_loader, device)
            print(f"Epoch {epoch}: Train Top1 Acc: {round(acc1 * 100, 2)}%")
        if EVAL_ON_TEST:
            print("Evaluating on test set...")
            acc1 = evaluate(model, representations, test_loader, device)
            print(f"Epoch {epoch}: Test Accuracy: {acc1 * 100:.2f}%")
        t1 = time.time_ns()
        print(f"Epoch time (sec): {(t1 - t0) / 10 ** 9:.1f}")

    if MODEL_IDENTITY:
        torch.save(model, 'models/MNIST_model_identity.pt')
    else:
        if WITHOUT_CLASS:
            torch.save(model, 'models/MNIST_model_without_class.pt')
        else:
            torch.save(model, 'models/MNIST_model.pt')

    print('Model params saved to disk.')
