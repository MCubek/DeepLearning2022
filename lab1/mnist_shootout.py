import random

import sklearn
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch import optim, nn

import pt_deep
import data

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters
input_size = 784  # 28x28
num_classes = 10
batch_size = 100
learning_rate = 0.001
regularization = 0.001


def load_mnist_data():
    dataset_root = '/tmp/mnist'  # change this to your preference
    mnist_train = torchvision.datasets.MNIST(dataset_root, train=True, transform=transforms.ToTensor(), download=True)
    mnist_test = torchvision.datasets.MNIST(dataset_root, train=False, transform=transforms.ToTensor(), download=True)

    return mnist_train, mnist_test


class MyDataLoader:
    def __init__(self, X, Y, batch_size, shuffle=True):
        size = X.shape[0]
        self.x_array = []
        self.y_array = []

        count = 0
        for i in range(batch_size, size, batch_size):
            self.x_array.append(X[count:i])
            self.y_array.append(Y[count:i])
            count = i

        if count <= size:
            self.x_array.append(X[count:size])
            self.y_array.append(Y[count:size])

    def shuffle(self):
        self.x_array, self.y_array = sklearn.utils.shuffle(self.x_array, self.y_array)

    def __len__(self):
        return len(self.y_array)

    def __iter__(self):
        if self.shuffle:
            self.shuffle()
        return iter(zip(self.x_array, self.y_array))

class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.input_size = input_size
        self.l1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        # no activation and no softmax at the end
        return out


def train(model, X, Y, batch_size, param_niter, param_delta, param_lambda):
    """Arguments:
       - X: model inputs [NxD], type: torch.Tensor
       - Yoh_: ground truth [NxC], type: torch.Tensor
       - param_niter: number of training iterations
       - param_delta: learning rate
    """

    # inicijalizacija optimizatora

    #model = NeuralNet(input_size, 500, num_classes).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=param_delta)
    criterion = nn.CrossEntropyLoss(label_smoothing=param_lambda)
    my_data_loader = MyDataLoader(X, Y, batch_size, shuffle=True)

    examples = iter(my_data_loader)
    example_data, example_targets = examples.__next__()
    for i in range(6):
        plt.subplot(2, 3, i + 1)
        plt.imshow(example_data[i], cmap='gray')
    plt.show()

    # petlja učenja
    n_total_steps = len(my_data_loader)
    for epoch in range(int(param_niter)):
        for i, (images, labels) in enumerate(my_data_loader):

            images = images.reshape(-1, 28 * 28).to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()

            if (i + 1) % 100 == 0:
                print(
                    f'Epoch [{epoch + 1}/{int(param_niter)}], Step [{i + 1}/{n_total_steps}], Loss: {loss.item():.4f}')

            optimizer.step()
            optimizer.zero_grad()


def eval(model, X, Y):
    with torch.no_grad():
        n_correct = 0
        n_samples = 0
        my_data_loader = MyDataLoader(X, Y, batch_size)
        for images, labels in my_data_loader:
            images = images.reshape(-1, 28 * 28).to(device)
            labels = labels.to(device)
            outputs = model(images)
            # max returns (value ,index)
            _, predicted = torch.max(outputs.data, 1)
            n_samples += labels.size(0)
            n_correct += (predicted == labels).sum().item()

        acc = 100.0 * n_correct / n_samples
        print(f'Accuracy of the network on the 10000 test images: {acc} %')


if __name__ == '__main__':
    mnist_train, mnist_test = load_mnist_data()

    model = pt_deep.PTDeep([input_size, 500, num_classes], torch.relu).to(device)

    x_train, y_train = mnist_train.data, mnist_train.targets
    x_test, y_test = mnist_test.data, mnist_test.targets

    x_train, x_test = x_train.float().div_(255.0), x_test.float().div_(255.0)

    train(model, x_train, y_train, batch_size=batch_size,
          param_niter=200, param_delta=0.01, param_lambda=1e-4)

    eval(model, x_test, y_test)

    weights = model.weights[0].detach().numpy().reshape(-1, 28, 28)

    fig, ax = plt.subplots(2, 5)

    for i, digit_weight in enumerate(weights):
        ax[i // 5 - 1, i % 5].imshow(digit_weight, cmap=plt.get_cmap('gray'))

    plt.show()
