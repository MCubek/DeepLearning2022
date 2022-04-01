import sys

import torch
import torch.nn as nn
from pathlib import Path
import os
import time
import math
import numpy as np
import matplotlib.pyplot as plt
import skimage as ski
import skimage.io
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

DATA_DIR = Path(__file__).parent / 'datasets' / 'MNIST'
SAVE_DIR = Path(__file__).parent / 'out' / 'torch' / 'MNIST'
RUNS_DIR = Path(__file__).parent / 'runs' / 'MNIST'

writer = SummaryWriter(str(RUNS_DIR))

num_epochs = 5
batch_size = 50
learning_rate_policy = {1: 1e-1, 3: 1e-2, 5: 1e-3, 7: 1e-4}
weight_decay = 1e-3
val_size = 5000


class CovolutionalModel(nn.Module):
    def __init__(self, in_channels, conv1_width, conv2_width, fc1_width, class_count):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, conv1_width, kernel_size=5, stride=1, padding=2, bias=True)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(conv1_width, conv2_width, kernel_size=5, stride=1, padding=2, bias=True)
        self.pool2 = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(conv2_width * 7 * 7, fc1_width, bias=True)
        self.fc_logits = nn.Linear(fc1_width, class_count, bias=True)

        # parametri su već inicijalizirani pozivima Conv2d i Linear
        # ali možemo ih drugačije inicijalizirati
        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear) and m is not self.fc_logits:
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)
        self.fc_logits.reset_parameters()

    def forward(self, x):
        h = self.conv1(x)
        h = self.pool1(h)
        h = torch.relu(h)
        h = self.conv2(h)
        h = self.pool2(h)
        h = torch.relu(h)
        h = h.view(h.shape[0], -1)
        h = self.fc1(h)
        h = torch.relu(h)
        logits = self.fc_logits(h)
        return logits


# noinspection DuplicatedCode
def draw_conv_filters(epoch, step, layer, save_dir):
    C = layer.C
    w = layer.weights.copy()
    num_filters = w.shape[0]
    k = int(np.sqrt(w.shape[1] / C))
    w = w.reshape(num_filters, C, k, k)
    w -= w.min()
    w /= w.max()
    border = 1
    cols = 8
    rows = math.ceil(num_filters / cols)
    width = cols * k + (cols - 1) * border
    height = rows * k + (rows - 1) * border
    # for i in range(C):
    for i in range(1):
        img = np.zeros([height, width])
        for j in range(num_filters):
            r = int(j / cols) * (k + border)
            c = int(j % cols) * (k + border)
            img[r:r + k, c:c + k] = w[j, i]
        filename = '%s_epoch_%02d_step_%06d_input_%03d.png' % (layer.name, epoch, step, i)
        ski.io.imsave(os.path.join(save_dir, filename), img)


def train(model, train_loader, val_loader):
    pass


# noinspection DuplicatedCode
def evaluate(name, data_loader, model, loss_function):
    print("\nRunning evaluation: ", name)
    num_examples = len(data_loader.dataset)
    assert num_examples % batch_size == 0
    num_batches = num_examples // batch_size
    cnt_correct = 0
    loss_avg = 0
    with torch.no_grad():
        for i, (images, labels) in enumerate(data_loader):
            logits = model.forward(images)
            loss_val = loss_function(logits, labels)
            loss_avg += loss_val

    valid_acc = cnt_correct / num_examples * 100
    loss_avg /= num_batches
    print(name + " accuracy = %.2f" % valid_acc)
    print(name + " avg loss = %.2f\n" % loss_avg)


if __name__ == '__main__':
    transform = transforms.ToTensor()
    train_dataset, test_dataset = [torchvision.datasets.MNIST(root=str(DATA_DIR),
                                                              train=True,
                                                              transform=transform,
                                                              download=True),
                                   torchvision.datasets.MNIST(root=str(DATA_DIR),
                                                              train=False,
                                                              transform=transform)]

    train_dataset, val_dataset = torch.utils.data.random_split(train_dataset,
                                                               [len(train_dataset.data) - val_size, val_size])

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True)

    val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                             batch_size=batch_size,
                                             shuffle=True)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=batch_size,
                                              shuffle=False)

    # Sample data
    example_data = iter(test_loader).__next__()[0]
    img_grid = torchvision.utils.make_grid(example_data)
    writer.add_image('mnist_images', img_grid)

    model = CovolutionalModel(1, 16, 32, 512, 10).to(device)
    writer.add_graph(model, example_data)

    train(model, train_loader, val_loader)
    evaluate('Test', test_loader, model, nn.CrossEntropyLoss())
