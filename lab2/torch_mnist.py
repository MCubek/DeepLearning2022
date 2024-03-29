import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

DATA_DIR = Path(__file__).parent / 'datasets' / 'MNIST'
RUNS_DIR = Path(__file__).parent / 'runs' / 'MNIST'

writer = SummaryWriter(str(RUNS_DIR))

num_epochs = 10
batch_size = 50
learning_rate_policy = {1: 1e-1, 3: 1e-2, 5: 1e-3, 7: 1e-4}
weight_decay = 1e-3
val_size = 5000


class CovolutionalModel(nn.Module):
    def __init__(self, in_channels, conv1_width, conv2_width, fc1_width, class_count):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, conv1_width, kernel_size=5, padding=2, bias=True)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(conv1_width, conv2_width, kernel_size=5, padding=2, bias=True)
        self.pool2 = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(conv2_width * 7 * 7, fc1_width, bias=True)
        self.fc_logits = nn.Linear(fc1_width, class_count, bias=True)

        # parametri su već inicijalizirani pozivima Conv2d i Linear
        # ali možemo ih drugačije inicijalizirati
        # self.reset_parameters()

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
        h = h.view((h.shape[0], -1))
        h = self.fc1(h)
        h = torch.relu(h)
        h = self.fc_logits(h)
        return h


# noinspection DuplicatedCode
def draw_conv_filters(epoch, step, layer):
    w = layer.weight.data.detach().cpu().numpy()
    num_filters, C, k = w.shape[:3]

    w = w.reshape(num_filters, C, k, k)
    w -= w.min()
    w /= w.max()
    border = 1
    cols = 8
    rows = math.ceil(num_filters / cols)
    width = cols * k + (cols - 1) * border
    height = rows * k + (rows - 1) * border
    for i in range(C):
        img = np.zeros([height, width]).astype(np.uint8)
        for j in range(num_filters):
            r = int(j / cols) * (k + border)
            c = int(j % cols) * (k + border)
            img[r:r + k, c:c + k] = w[j, i] * 255
        filename = '%s_epoch_%02d_step_%06d_input_%03d' % ('mnist_conv1', epoch, step, i)
        grid = torchvision.utils.make_grid(torch.from_numpy(img))
        writer.add_image(filename, grid, step)


# noinspection DuplicatedCode
def train(model, train_loader, val_loader):
    criterion = nn.CrossEntropyLoss()
    optimizer = None

    running_loss = 0.0
    running_correct = 0
    n_total_steps = len(train_loader)
    n_dataset_size = len(train_loader.dataset)

    for epoch in range(num_epochs):
        running_correct_epoch = 0

        if (epoch + 1) in learning_rate_policy:
            optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate_policy[epoch + 1],
                                        weight_decay=weight_decay)
        assert optimizer is not None

        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            predicted_correct = (predicted == labels).sum().item()
            running_correct += predicted_correct
            running_correct_epoch += predicted_correct

            if i % 10 == 0:
                print(
                    f'Epoch [{epoch + 1}/{num_epochs}], Step [{i * batch_size}/{n_dataset_size}], Loss: {loss.item():.4f}')

            if i % 100 == 0:
                writer.add_scalar('loss/train batch', running_loss / 100, epoch * n_total_steps + i)

                running_accuracy = running_correct / 100 / predicted.size(0)
                print("Train accuracy on batch = %.4f" % running_accuracy)
                writer.add_scalar('accuracy/train batch', running_accuracy, epoch * n_total_steps + i)

                running_loss = 0.0
                running_correct = 0

                draw_conv_filters(epoch + 1, i * batch_size, model.conv1)

        epoch_accuracy = running_correct_epoch / n_dataset_size * 100
        print("Train accuracy after epoch = %.4f" % epoch_accuracy)
        writer.add_scalar('accuracy/train epoch', epoch_accuracy, epoch)

        valid_acc, valid_loss_avg = evaluate("Validation", val_loader, model, criterion)
        writer.add_scalar('accuracy/validate', valid_acc, epoch)
        writer.add_scalar('loss/validate', valid_loss_avg, epoch)

    return model


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
            images = images.to(device)
            labels = labels.to(device)

            outputs = model.forward(images)
            loss_val = loss_function(outputs, labels)
            loss_avg += loss_val

            _, predicted = torch.max(outputs.data, 1)
            cnt_correct += (predicted == labels).sum().item()

    valid_acc = cnt_correct / num_examples * 100
    loss_avg /= num_batches

    print(name + " accuracy = %.2f" % valid_acc)
    print(name + " avg loss = %.2f\n" % loss_avg)

    return valid_acc, loss_avg


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

    model = CovolutionalModel(1, 16, 32, 512, 10)
    writer.add_graph(model, example_data)

    model = model.to(device)
    train(model, train_loader, val_loader)

    class_preds, class_labels = evaluate('Test', test_loader, model, nn.CrossEntropyLoss())

    writer.close()
