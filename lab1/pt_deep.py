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

        s = X.double()
        for weight, bias in zip(self.weights, self.biases):
            s = s.mm(weight.double()) + bias
            s = self.activation_fun(s)

        prob = torch.softmax(s, dim=1)

        return prob

    def get_loss(self, X, Yoh_):
        prob = self.forward(X)

        log_prob = torch.log(prob) * Yoh_
        log_sum = torch.sum(log_prob, dim=1)
        log_prob_mean = torch.mean(log_sum)

        return -log_prob_mean

    def count_params(self):
        for name, param, in self.named_parameters():
            print(f'{name}:{param}\n')


def train(model, X, Yoh_, param_niter, param_delta, param_lambda):
    """Arguments:
       - X: model inputs [NxD], type: torch.Tensor
       - Yoh_: ground truth [NxC], type: torch.Tensor
       - param_niter: number of training iterations
       - param_delta: learning rate
    """

    # inicijalizacija optimizatora
    optimizer = optim.SGD(model.parameters(), lr=param_delta)

    # petlja učenja
    # ispisujte gubitak tijekom učenja
    for i in range(int(param_niter)):
        loss = model.get_loss(X, Yoh_) + param_lambda * np.sum(
            [torch.norm(weight).detach().numpy() for weight in model.weights])

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
    res = model.forward(X).detach().numpy()

    return np.argmax(res, axis=1)
    # koristite torch.Tensor.detach() i torch.Tensor.numpy()


if __name__ == "__main__":
    # inicijaliziraj generatore slučajnih brojeva
    torch.manual_seed(100)
    np.random.seed(100)

    # instanciraj podatke X i labele Yoh_
    X, Y_ = data.sample_gmm_2d(4, 2, 40)

    Yoh_ = data.class_to_onehot(Y_)

    X_tensor = torch.from_numpy(X)
    Yoh_ = torch.from_numpy(Yoh_)

    # definiraj model:
    ptdeep = PTDeep([2, 10, 10, 2], torch.relu)
    ptdeep.count_params()

    # nauči parametre (X i Yoh_ moraju biti tipa torch.Tensor):
    train(ptdeep, X_tensor, Yoh_, param_niter=1e4, param_delta=0.1, param_lambda=1e-4)

    # dohvati vjerojatnosti na skupu za učenje
    probs = eval(ptdeep, X_tensor)

    # ispiši performansu (preciznost i odziv po razredima)

    acc, precission_recall, conf_matrix = data.eval_perf_multi(probs, Y_)

    print(f'accuracy:{acc}\nprecission and recall per class:{precission_recall}\nconfusion matrix:{conf_matrix}')

    # iscrtaj rezultate, decizijsku plohu
    rect = (np.min(X, axis=0), np.max(X, axis=0))
    data.graph_surface(lambda x: eval(ptdeep, torch.from_numpy(x)), rect, offset=0)

    # graph the data points
    data.graph_data(X, Y_, probs, special=[])

    plt.show()
