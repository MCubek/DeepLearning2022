import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt


def predict(x, a, b):
    return a * x + b


def lin_reg(X, Y, param_niter=1e4, param_delta=0.001):
    print_freq = 1000

    a = torch.randn(1, requires_grad=True)
    b = torch.zeros(1, requires_grad=True)

    X_tensor = torch.tensor(X)
    Y_tensor = torch.tensor(Y)

    optimizer = optim.SGD([a, b], lr=param_delta)

    for i in range(int(param_niter)):
        # afin regresijski model
        Y_ = predict(X_tensor, a, b)

        diff = (Y_tensor - Y_)

        # kvadratni gubitak
        loss = torch.mean(diff ** 2)

        # računanje gradijenata
        loss.backward()

        # korak optimizacije
        optimizer.step()

        if i % print_freq == 0:
            print(f'step: {i}, a.grad:{a.grad.item():.4f}, b.grad:{b.grad.item():.4f}')
            print(f'a.grad:{-2 * torch.mean(diff * X_tensor):.4f}, b.grad:{-2 * torch.mean(diff):.4f}')

        # Postavljanje gradijenata na nulu
        optimizer.zero_grad()

        if i % print_freq == 0:
            print(f'loss:{loss}, a:{a.item():.4f}, b {b.item():.4f}\n')

    return a.item(), b.item()


if __name__ == '__main__':
    X = [1, 2, 5, 7, 10]
    Y = [200, 290, 600, 801, 1000]

    a, b = lin_reg(X, Y)

    y_predicted = [predict(x, a, b) for x in X]

    plt.plot(X, Y, 'ro')
    plt.plot(X, y_predicted, 'b')
    plt.show()
