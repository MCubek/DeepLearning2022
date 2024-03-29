import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

DATA_DIR = Path(__file__).parent / 'datasets' / 'CIFAR'
RUNS_DIR = Path(__file__).parent / 'runs' / 'CIFAR'

writer = SummaryWriter(str(RUNS_DIR))

num_epochs = 100
batch_size = 50
learning_rate = 1e-3
weight_decay = 5e-4
betas = (0.9, 0.995)
gamma_param = 0.99
val_size = 5000

mean = (0.4913997551666284, 0.48215855929893703, 0.4465309133731618)
std = (0.24703225141799082, 0.24348516474564, 0.26158783926049628)

norm = transforms.Normalize(mean, std)
unnorm = transforms.Normalize(
    mean=[-m / s for m, s in zip(mean, std)],
    std=[1 / s for s in std]
)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


def calculate_conv_output(image_width, pool_kernel, pool_stride):
    return (image_width - pool_kernel) // pool_stride + 1


class CovolutionalCifarModel(nn.Module):
    def __init__(self, in_channels, conv1_width, conv2_width, fc1_width, fc2_width, class_count, pool_kernel,
                 pool_stride, image_width):
        super().__init__()

        self.convolution = nn.Sequential(
            nn.Conv2d(in_channels, conv1_width, kernel_size=5, padding=2, bias=True),
            nn.MaxPool2d(pool_kernel, pool_stride),
            nn.ReLU(inplace=True),
            nn.Conv2d(conv1_width, conv2_width, kernel_size=5, padding=2, bias=True),
            nn.MaxPool2d(pool_kernel, pool_stride),
            nn.ReLU(inplace=True)
        )

        for _ in range(2):
            image_width = calculate_conv_output(image_width, pool_kernel, pool_stride)

        self.fc = nn.Sequential(
            nn.Linear(conv2_width * image_width ** 2, fc1_width, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(fc1_width, fc2_width, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(fc2_width, class_count, bias=True)
        )

    def forward(self, x):
        h = self.convolution(x)
        h = torch.flatten(h, 1)
        h = self.fc(h)
        return h


# noinspection DuplicatedCode
def draw_conv_filters(epoch, step, layer):
    w = layer.weight.data.detach().cpu().numpy()
    num_filters, num_channels, k = w.shape[:3]

    assert w.shape[3] == w.shape[2]
    w = w.transpose(2, 3, 1, 0)
    w -= w.min()
    w /= w.max()
    border = 1
    cols = 8
    rows = math.ceil(num_filters / cols)
    width = cols * k + (cols - 1) * border
    height = rows * k + (rows - 1) * border
    img = np.zeros([height, width, num_channels])
    for i in range(num_filters):
        r = int(i / cols) * (k + border)
        c = int(i % cols) * (k + border)
        img[r:r + k, c:c + k, :] = w[:, :, :, i]
    filename = '%s_epoch_%02d_step_%06d' % ('cifar_conv1', epoch, step)
    img = np.moveaxis(img, -1, 0)
    img = (img * 255).astype(np.uint8)
    grid = torchvision.utils.make_grid(torch.from_numpy(img))
    writer.add_image(filename, grid, step)


# noinspection DuplicatedCode
def train(model, train_loader, val_loader):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay, betas=betas)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=gamma_param)
    criterion = nn.CrossEntropyLoss()

    running_loss = 0.0
    running_correct = 0
    n_total_steps = len(train_loader)
    n_dataset_size = len(train_loader.dataset)

    # Filters first
    draw_conv_filters(0, 0, model.convolution[0])

    new_lr = learning_rate

    for epoch in range(num_epochs):
        writer.add_scalar('learning rate', new_lr, epoch)

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

                draw_conv_filters(epoch + 1, i * batch_size, model.convolution[0])

        scheduler.step()

        valid_acc, valid_loss_avg, M = evaluate("Training after epoch", train_loader, model, criterion)
        writer.add_scalar('accuracy/train epoch', valid_acc, epoch)
        writer.add_scalar('loss/train epoch', valid_loss_avg, epoch)

        valid_acc, valid_loss_avg, M = evaluate("Validation", val_loader, model, criterion)
        writer.add_scalar('accuracy/validate', valid_acc, epoch)
        writer.add_scalar('loss/validate', valid_loss_avg, epoch)

        new_lr = scheduler.get_last_lr()[-1]
        print(f'New Learning rate = {new_lr}.')

    return model


def eval_perf_multi(Y, Y_):
    pr = []
    n = max(Y_) + 1
    M = np.bincount(n * Y_ + Y, minlength=n * n).reshape(n, n)
    for i in range(n):
        tp_i = M[i, i]
        fn_i = np.sum(M[i, :]) - tp_i
        fp_i = np.sum(M[:, i]) - tp_i
        tn_i = np.sum(M) - fp_i - fn_i - tp_i
        recall_i = tp_i / (tp_i + fn_i)
        precision_i = tp_i / (tp_i + fp_i)
        pr.append((recall_i, precision_i))

    accuracy = np.trace(M) / np.sum(M)

    return accuracy, pr, M


# noinspection DuplicatedCode
def evaluate(name, data_loader, model, loss_function):
    print("\nRunning evaluation: ", name)
    num_examples = len(data_loader.dataset)
    assert num_examples % batch_size == 0

    num_batches = num_examples // batch_size
    loss_avg = 0

    predicted_eval = []
    true_eval = []

    with torch.no_grad():
        for i, (images, labels) in enumerate(data_loader):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model.forward(images)
            loss_val = loss_function(outputs, labels)
            loss_avg += loss_val

            _, predicted = torch.max(outputs.data, 1)

            predicted_eval.append(predicted.detach().cpu().numpy())
            true_eval.append(labels.detach().cpu().numpy())

        predicted_eval = np.reshape(predicted_eval, -1)
        true_eval = np.reshape(true_eval, -1)
        acc, pr, M = eval_perf_multi(predicted_eval, true_eval)
        loss_avg /= num_batches

        print(name + " accuracy = %.4f" % (acc * 100))
        print(name + " avg loss = %.4f\n" % loss_avg)

        return acc * 100, loss_avg, M


def draw_image(img, title):
    img = unnorm(img).numpy()
    img = (img * 255).astype(np.uint8)
    grid = torchvision.utils.make_grid(torch.from_numpy(img))
    writer.add_image(title, grid)


def print_20_highest_loss(model, data_loader):
    data = None
    loss = None
    predicted = None
    true = None

    loss_function = nn.CrossEntropyLoss(reduction='none')

    with torch.no_grad():
        for i, (images, labels) in enumerate(data_loader):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model.forward(images)
            loss_val = loss_function(outputs, labels)

            if data is None:
                data = images.detach().cpu()
                loss = loss_val.detach().cpu()
                predicted = outputs.detach().cpu()
                true = labels.detach().cpu()
            else:
                data = torch.cat((data, images.detach().cpu()), 0)
                loss = torch.cat((loss, loss_val.detach().cpu()), 0)
                predicted = torch.cat((predicted, outputs.detach().cpu()), 0)
                true = torch.cat((true, labels.detach().cpu()), 0)

        top_i = torch.topk(loss, 20)[1].detach().numpy()

        for count, i in enumerate(top_i):
            draw_image(data[i],
                       f'Worst No{count}: True: {classes[true[i].data]}, Predicted: {list(classes[x] for x in torch.topk(predicted[i, :], 3)[1].data.tolist())}.')


def print_top_3(M):
    m_range = range(len(M[0, :]))

    acc = []
    for i in m_range:
        acc.append((i, M[i, i]))

    acc.sort(key=lambda x: x[1], reverse=True)

    for i, item in enumerate(acc[:3]):
        print(f'Top item number {i + 1} is {classes[item[0]]}.')


if __name__ == '__main__':
    train_transform = transforms.Compose(
        [transforms.ToTensor(), transforms.RandomHorizontalFlip(), transforms.RandomRotation(degrees=10),
         transforms.ColorJitter(brightness=0.5), norm])
    test_transform = transforms.Compose(
        [transforms.ToTensor(), norm])
    train_dataset, test_dataset = [torchvision.datasets.CIFAR10(root=str(DATA_DIR),
                                                                train=True,
                                                                transform=train_transform,
                                                                download=True),
                                   torchvision.datasets.CIFAR10(root=str(DATA_DIR),
                                                                train=False,
                                                                transform=test_transform)]

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

    test_loader_no_batch = torch.utils.data.DataLoader(dataset=test_dataset,
                                                       batch_size=len(test_dataset),
                                                       shuffle=False)

    # Sample data
    example_data = iter(test_loader).__next__()[0]
    draw_image(example_data.detach(), 'cifar_images')

    model = CovolutionalCifarModel(3, 16, 32, 256, 128, 10, pool_kernel=3, pool_stride=2, image_width=32)
    writer.add_graph(model, example_data)

    model = model.to(device)

    train(model, train_loader, val_loader)

    class_preds, class_labels, M = evaluate('Test', test_loader, model, nn.CrossEntropyLoss())

    print_20_highest_loss(model, test_loader_no_batch)

    print_top_3(M)

    writer.close()
