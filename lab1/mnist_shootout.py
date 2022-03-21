import matplotlib.pyplot as plt
import numpy as np
import sklearn
import torch
import torchvision
import torchvision.transforms as transforms
from torch import nn

import data
import ksvm_wrap
import pt_deep

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters
input_size = 784  # 28x28
num_classes = 10
batch_size = 100
num_epoch = 20
learning_rate = 0.01
regularization_factor = 0.005
configurations = [[input_size, num_classes], [input_size, 100, num_classes], [input_size, 100, 100, num_classes],
                  [input_size, 100, 100, 100, num_classes]]


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


def train(model, data_loader, param_niter, param_delta, param_lambda, log=True):
    """Arguments:
       - X: model inputs [NxD], type: torch.Tensor
       - Yoh_: ground truth [NxC], type: torch.Tensor
       - param_niter: number of training iterations
       - param_delta: learning rate
    """

    # inicijalizacija optimizatora

    # model = NeuralNet(input_size, 500, num_classes).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=param_delta)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=1 - param_delta)
    criterion = nn.CrossEntropyLoss(label_smoothing=param_lambda)

    # petlja učenja
    loss = None
    for epoch in range(int(param_niter)):
        for images, labels in data_loader:
            images = images.reshape(-1, 28 * 28).to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

        if log:
            print(f'Epoch [{epoch + 1}/{int(param_niter)}], Loss: {loss.item():.4f}')
        scheduler.step()


def eval(model, X, y_true):
    with torch.no_grad():
        y_predicted = model(X.reshape(-1, 28 * 28)).detach().numpy()
        y_predicted = np.argmax(y_predicted, axis=1)
        acc, precission_recall, conf_matrix = data.eval_perf_multi(y_predicted, y_true)

        print(f'accuracy:{acc}\nprecission and recall per class:{precission_recall}\nconfusion matrix:\n{conf_matrix}')


def plot_sample(data_loader):
    examples = iter(data_loader)
    example_data, example_targets = examples.__next__()
    for i in range(6):
        plt.subplot(2, 3, i + 1)
        plt.imshow(example_data[i], cmap='gray')
    plt.show()


def get_one_loss(model, data_loader):
    with torch.no_grad():
        images, labels = iter(data_loader).__next__()
        images = images.reshape(-1, 28 * 28).to(device)
        labels = labels.to(device)

        outputs = model(images)
        criterion = nn.CrossEntropyLoss(label_smoothing=regularization_factor)
        return criterion(outputs, labels)


def plot_weights(model):
    weights = model.weights[0].detach().numpy().reshape(-1, 28, 28)

    fig, ax = plt.subplots(2, 5)

    for i, digit_weight in enumerate(weights):
        ax[i // 5 - 1, i % 5].imshow(digit_weight.T, cmap=plt.get_cmap('gray'))

    plt.show()


if __name__ == '__main__':
    mnist_train, mnist_test = load_mnist_data()

    x_train, y_train = mnist_train.data, mnist_train.targets
    x_test, y_test = mnist_test.data, mnist_test.targets
    x_train, x_test = x_train.float().div_(255.0), x_test.float().div_(255.0)

    my_data_loader_train = MyDataLoader(x_train, y_train, batch_size, shuffle=True)
    my_data_loader_test = MyDataLoader(x_train, y_train, batch_size, shuffle=True)

    # Plot digit sample
    plot_sample(my_data_loader_train)

    # 8)
    model = pt_deep.PTDeep([input_size, 100, num_classes], torch.relu).to(device)
    loss = get_one_loss(model, my_data_loader_train)
    print(f"Loss before training = {loss}")

    # 1)
    print(f'Regularization parameters:')
    regularization_params = [0.001, 0.005, 0.01]
    for lambda_param in regularization_params:
        print(f'param={lambda_param}')

        model = pt_deep.PTDeep([input_size, num_classes], torch.relu).to(device)
        train(model, my_data_loader_train,
              param_niter=num_epoch, param_delta=learning_rate, param_lambda=lambda_param, log=True)
        eval(model, x_test, y_test)
        plot_weights(model)

    # 2)
    for i, config in enumerate(configurations):
        print(f'\nConfiguration = {config}')
        model = pt_deep.PTDeep(config, torch.relu).to(device)

        train(model, my_data_loader_train,
              param_niter=num_epoch, param_delta=learning_rate, param_lambda=regularization_factor)

        eval(model, x_test, y_test)

    # 9)
    svm_kernels = ['linear', 'rbf']
    for kernel in svm_kernels:
        print(f'SVM Kernel = {kernel}.')
        model = ksvm_wrap.KSVMWrap(x_train.reshape(-1, 28 * 28), y_train, param_kernel=kernel)
        y_predicted = model.predict(x_test.reshape(-1, 28 * 28))
        acc, precission_recall, conf_matrix = data.eval_perf_multi(y_predicted, y_test)

        print(f'accuracy:{acc}\nprecission and recall per class:{precission_recall}\nconfusion matrix:\n{conf_matrix}')
