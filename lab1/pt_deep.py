import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import data


class PTDeep(nn.Module):
    def __init__(self, param_sizes, param_activation_fun):
        """Arguments:
           - D: dimensions of each datapoint
           - C: number of classes
        """
        super().__init__()

        # inicijalizirati parametre (koristite nn.Parameter):
        self.weights = nn.ParameterList(
            [nn.Parameter(torch.randn(param_sizes[i - 1], param_sizes[i])) for i in range(1, len(param_sizes))])
        self.biases = nn.ParameterList([nn.Parameter(torch.zeros(size)) for size in param_sizes[1:]])

        self.activation_fun = param_activation_fun

    def forward(self, X):
        # unaprijedni prolaz modela: izračunati vjerojatnosti

        s = X
        params = len(self.weights)
        for i, (weight, bias) in enumerate(zip(self.weights, self.biases)):
            s = s.mm(weight) + bias
            if i < params - 1:
                s = self.activation_fun(s)

        return s

    def count_params(self):
        for name, param, in self.named_parameters():
            print(f'{name}:{param}\n')


def train(model, X, Y, param_niter, param_delta, param_lambda):
    """Arguments:
       - X: model inputs [NxD], type: torch.Tensor
       - Yoh_: ground truth [NxC], type: torch.Tensor
       - param_niter: number of training iterations
       - param_delta: learning rate
    """

    # inicijalizacija optimizatora
    optimizer = optim.SGD(model.parameters(), lr=param_delta)
    criterion = nn.CrossEntropyLoss(label_smoothing=param_lambda)

    # petlja učenja
    # ispisujte gubitak tijekom učenja
    for i in range(int(param_niter)):
        y_pred = model(X)
        loss = criterion(y_pred, Y)

        loss.backward()

        if i % 100 == 0:
            print(f'iter: {i}, loss:{loss}')

        optimizer.step()
        optimizer.zero_grad()


def eval(model, X):
    """Arguments:
       - model: type: PTLogreg
       - X: actual datapoints [NxD], type: np.array
       Returns: predicted class probabilites [NxC], type: np.array
    """
    # ulaz je potrebno pretvoriti u torch.Tensor
    # izlaze je potrebno pretvoriti u numpy.array
    with torch.no_grad():
        y_predicted = model(X).detach().numpy()
        return np.argmax(y_predicted, axis=1)
    # koristite torch.Tensor.detach() i torch.Tensor.numpy()


if __name__ == "__main__":
    torch.manual_seed(100)
    np.random.seed(100)

    X, Y_ = data.sample_gmm_2d(4, 2, 40)

    X_tensor = torch.from_numpy(X.astype(np.float32))
    Y_tensor = torch.from_numpy(Y_.astype(np.int64))

    configs = [[2, 2], [2, 10, 2], [2, 10, 10, 2]]

    for config in configs:
        print(f'Config: {config}\n')

        ptdeep = PTDeep(config, torch.relu)
        ptdeep.count_params()

        train(ptdeep, X_tensor, Y_tensor, param_niter=1e4, param_delta=0.1, param_lambda=1e-4)

        y_predicted = eval(ptdeep, X_tensor)

        # ispiši performansu (preciznost i odziv po razredima)
        acc, precission_recall, conf_matrix = data.eval_perf_multi(y_predicted, Y_)

        print(f'accuracy:{acc}\nprecission and recall per class:{precission_recall}\nconfusion matrix:\n{conf_matrix}')

        # iscrtaj rezultate, decizijsku plohu
        rect = (np.min(X, axis=0), np.max(X, axis=0))
        data.graph_surface(lambda x: eval(ptdeep, torch.from_numpy(x.astype(np.float32))), rect, offset=0)
        data.graph_data(X, Y_, y_predicted, special=[])

        plt.show()
