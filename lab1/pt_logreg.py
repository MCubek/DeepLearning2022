import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import data


class PTLogreg(nn.Module):
    def __init__(self, D, C):
        """Arguments:
           - D: dimensions of each datapoint
           - C: number of classes
        """
        super().__init__()

        # inicijalizirati parametre (koristite nn.Parameter):
        self.W = nn.Parameter(torch.randn(D, C), requires_grad=True)
        self.b = nn.Parameter(torch.randn(C), requires_grad=True)

    def forward(self, X):
        # unaprijedni prolaz modela: izračunati vjerojatnosti

        return X.mm(self.W) + self.b


def train(model, X, Y, param_niter, param_delta, param_lambda):
    """Arguments:
       - X: model inputs [NxD], type: torch.Tensor
       - Yoh_: ground truth [NxC], type: torch.Tensor
       - param_niter: number of training iterations
       - param_delta: learning rate
    """

    # inicijalizacija optimizatora
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=param_delta)

    # petlja učenja
    # ispisujte gubitak tijekom učenja
    for i in range(int(param_niter)):
        output = model(X)
        loss = criterion(output, Y) + param_lambda * torch.linalg.norm(model.W)

        loss.backward()

        if i % 100 == 0:
            print(f'iter: {i}, loss:{loss}')

        optimizer.step()

        optimizer.zero_grad()


def eval(model, X):
    """Arguments:
       - model: type: PTLogreg
       - X: actual datapoints [NxD], type: np.array
       Return: predicted class probabilites [NxC], type: np.array
    """
    # ulaz je potrebno pretvoriti u torch.Tensor
    # izlaze je potrebno pretvoriti u numpy.array
    with torch.no_grad():
        y_predicted = model(X).detach().numpy()
        return np.argmax(y_predicted, axis=1)

    # koristite torch.Tensor.detach() i torch.Tensor.numpy()


if __name__ == "__main__":
    # inicijaliziraj generatore slučajnih brojeva
    torch.manual_seed(100)
    np.random.seed(100)

    # instanciraj podatke X i labele Yoh_
    X, Y_ = data.sample_gauss_2d(3, 100)

    X_tensor = torch.from_numpy(X.astype("float32"))
    Y_tensor = torch.from_numpy(Y_.astype("int64"))

    # definiraj model:
    model_logreg = PTLogreg(X.shape[1], np.unique(Y_).size)

    train(model_logreg, X_tensor, Y_tensor, 10000, 0.3, 0.01)

    y_pred = eval(model_logreg, X_tensor)

    # ispiši performansu (preciznost i odziv po razredima)
    acc, precission_recall, conf_matrix = data.eval_perf_multi(y_pred, Y_)

    print(f'accuracy:{acc}\nprecission and recall per class:{precission_recall}\nconfusion matrix:\n{conf_matrix}')

    # iscrtaj rezultate, decizijsku plohu
    rect = (np.min(X, axis=0), np.max(X, axis=0))
    data.graph_surface(lambda x: eval(model_logreg, torch.from_numpy(x.astype("float32"))), rect, offset=0)

    # graph the data points
    data.graph_data(X, Y_, y_pred, special=[])

    plt.show()
