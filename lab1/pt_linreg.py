import torch
import torch.nn as nn
import torch.optim as optim


def lin_reg(X, Y, param_niter=1e4, param_delta=0.001):
    print_freq = 100

    a = torch.randn(1, requires_grad=True)
    b = torch.randn(1, requires_grad=True)

    X_tensor = torch.tensor(X)
    Y_tensor = torch.tensor(Y)

    optimizer = optim.SGD([a, b], lr=param_delta)

    for i in range(int(param_niter)):
        # afin regresijski model
        Y_ = a * X_tensor + b

        diff = (Y_tensor - Y_)

        # kvadratni gubitak
        loss = torch.mean(diff ** 2)

        # računanje gradijenata
        loss.backward()

        # korak optimizacije
        optimizer.step()

        if i % print_freq == 0:
            print(f'step: {i}, a.grad:{a.grad}, b.grad:{b.grad}')
            print(f'a.grad{-2 * torch.mean(diff * X_tensor)}, b.grad{-2 * torch.mean(diff)}')

        # Postavljanje gradijenata na nulu
        optimizer.zero_grad()

        if i % print_freq == 0:
            print(f'loss:{loss}, Y_:{Y_.data}, a:{a.data}, b {b.data}\n')


if __name__ == '__main__':
    X = [1, 2, 5, 7, 10]
    Y = [200, 290, 600, 801, 1101]

    lin_reg(X, Y)
