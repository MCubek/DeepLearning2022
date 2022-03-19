import torch
import torchvision
import matplotlib.pyplot as plt

import pt_deep
import data


def load_mnist_data():
    dataset_root = '/tmp/mnist'  # change this to your preference
    mnist_train = torchvision.datasets.MNIST(dataset_root, train=True, download=True)
    mnist_test = torchvision.datasets.MNIST(dataset_root, train=False, download=True)

    return mnist_train, mnist_test


if __name__ == '__main__':
    mnist_train, mnist_test = load_mnist_data()

    x_train, y_train = mnist_train.data, mnist_train.targets
    x_test, y_test = mnist_test.data, mnist_test.targets

    x_train, x_test = x_train.float().div_(255.0), x_test.float().div_(255.0)

    N = x_train.shape[0]
    D = x_train.shape[1] * x_train.shape[2]
    C = y_train.max().add_(1).item()

    x_train, x_test = x_train.view(-1, D), x_test.view(-1, D)

    y_train = torch.from_numpy(data.class_to_onehot(y_train.detach().numpy()))
    y_test = torch.from_numpy(data.class_to_onehot(y_test.detach().numpy()))

    deep_model = pt_deep.PTDeep([D, C], torch.relu)
    pt_deep.train(deep_model, x_train, y_train, param_niter=1000, param_delta=0.1, param_lambda=1e-4)

    weights = deep_model.weights[0].detach().numpy().reshape(-1, 28, 28)

    fig, ax = plt.subplots(2, 5)

    for i, digit_weight in enumerate(weights):
        ax[i // 5 - 1, i % 5].imshow(digit_weight, cmap=plt.get_cmap('gray'))

    plt.show()
