import numpy as np
import matplotlib.pyplot as plt
import random

import pdb
import IPython


class Random2DGaussian:
    """Random bivariate normal distribution sampler

    Hardwired parameters:
        d0min,d0max: horizontal range for the mean
        d1min,d1max: vertical range for the mean
        scalecov: controls the covariance range

    Methods:
        __init__: creates a new distribution

        get_sample(n): samples n datapoints

    """

    d0min = 0
    d0max = 10
    d1min = 0
    d1max = 10
    scalecov = 5

    def __init__(self):
        dw0, dw1 = self.d0max - self.d0min, self.d1max - self.d1min
        mean = (self.d0min, self.d1min)
        mean += np.random.random_sample(2) * (dw0, dw1)
        eigvals = np.random.random_sample(2)
        eigvals *= (dw0 / self.scalecov, dw1 / self.scalecov)
        eigvals **= 2
        theta = np.random.random_sample() * np.pi * 2
        R = [[np.cos(theta), -np.sin(theta)],
             [np.sin(theta), np.cos(theta)]]
        Sigma = np.dot(np.dot(np.transpose(R), np.diag(eigvals)), R)
        self.get_sample = lambda n: np.random.multivariate_normal(mean, Sigma, n)


def graph_surface(function, rect, offset=0.5, width=256, height=256):
    """Creates a surface plot (visualize with plt.show)

    Arguments:
      function: surface to be plotted
      rect:     function domain provided as:
                ([x_min,y_min], [x_max,y_max])
      offset:   the level plotted as a contour plot

    Returns:
      None
    """

    lsw = np.linspace(rect[0][1], rect[1][1], width)
    lsh = np.linspace(rect[0][0], rect[1][0], height)
    xx0, xx1 = np.meshgrid(lsh, lsw)
    grid = np.stack((xx0.flatten(), xx1.flatten()), axis=1)

    # get the values and reshape them
    values = function(grid).reshape((width, height))

    # fix the range and offset
    delta = offset if offset else 0
    maxval = max(np.max(values) - delta, - (np.min(values) - delta))

    # draw the surface and the offset
    plt.pcolormesh(xx0, xx1, values,
                   vmin=delta - maxval, vmax=delta + maxval)

    if offset != None:
        plt.contour(xx0, xx1, values, colors='black', levels=[offset])


def graph_data(X, Y_, Y, special=[]):
    """Creates a scatter plot (visualize with plt.show)

    Arguments:
        X:       datapoints
        Y_:      groundtruth classification indices
        Y:       predicted class indices
        special: use this to emphasize some points

    Returns:
        None
    """
    # colors of the datapoint markers
    palette = ([0.5, 0.5, 0.5], [1, 1, 1], [0.2, 0.2, 0.2])
    colors = np.tile([0.0, 0.0, 0.0], (Y_.shape[0], 1))
    for i in range(len(palette)):
        colors[Y_ == i] = palette[i]

    # sizes of the datapoint markers
    sizes = np.repeat(20, len(Y_))
    sizes[special] = 40

    # draw the correctly classified datapoints
    good = (Y_ == Y)
    plt.scatter(X[good, 0], X[good, 1], c=colors[good],
                s=sizes[good], marker='o')

    # draw the incorrectly classified datapoints
    bad = (Y_ != Y)
    plt.scatter(X[bad, 0], X[bad, 1], c=colors[bad],
                s=sizes[bad], marker='s')


def class_to_onehot(Y):
    Yoh = np.zeros((len(Y), max(Y) + 1))
    Yoh[range(len(Y)), Y] = 1
    return Yoh


def eval_perf_binary(Y, Y_):
    tp = sum(np.logical_and(Y == Y_, Y_ == True))
    fn = sum(np.logical_and(Y != Y_, Y_ == True))
    tn = sum(np.logical_and(Y == Y_, Y_ == False))
    fp = sum(np.logical_and(Y != Y_, Y_ == False))
    recall = tp / (tp + fn)
    precision = tp / (tp + fp)
    accuracy = (tp + tn) / (tp + fn + tn + fp)
    return accuracy, recall, precision


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


def eval_AP(ranked_labels):
    """Recovers AP from ranked labels"""

    n = len(ranked_labels)
    pos = sum(ranked_labels)
    neg = n - pos

    tp = pos
    tn = 0
    fn = 0
    fp = neg

    sumprec = 0
    # IPython.embed()
    for x in ranked_labels:
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)

        if x:
            sumprec += precision

        # print (x, tp,tn,fp,fn, precision, recall, sumprec)
        # IPython.embed()

        tp -= x
        fn += x
        fp -= not x
        tn += not x

    return sumprec / pos


def sample_gauss_2d(nclasses, nsamples):
    # create the distributions and groundtruth labels
    Gs = []
    Ys = []
    for i in range(nclasses):
        Gs.append(Random2DGaussian())
        Ys.append(i)

    # sample the dataset
    X = np.vstack([G.get_sample(nsamples) for G in Gs])
    Y_ = np.hstack([[Y] * nsamples for Y in Ys])

    return X, Y_


def sample_gmm_2d(ncomponents, nclasses, nsamples):
    # create the distributions and groundtruth labels
    Gs = []
    Ys = []
    for i in range(ncomponents):
        Gs.append(Random2DGaussian())
        Ys.append(np.random.randint(nclasses))

    # sample the dataset
    X = np.vstack([G.get_sample(nsamples) for G in Gs])
    Y_ = np.hstack([[Y] * nsamples for Y in Ys])

    return X, Y_


def myDummyDecision(X):
    scores = X[:, 0] + X[:, 1] - 5
    return scores


param_niter = 100
param_delta = 1e-2


def binlogreg_train(X, Y_):
    """
      Argumenti
        X:  podatci, np.array NxD
        Y_: indeksi razreda, np.array Nx1

      Povratne vrijednosti
        w, b: parametri logističke regresije
    """
    N, D = X.shape
    w = np.random.randn(D, 1)
    b = 0

    for i in range(param_niter):
        scores = np.dot(X, w) + b

        exp_s = np.exp(scores)
        probs = exp_s / (1 + exp_s)

        loss = -np.sum(np.log(probs)) / N

        if i % 10 == 0:
            print("iteration {}: loss {}".format(i, loss))

        dL_dscores = probs - (Y_ == 1).astype(int)

        grad_w = np.transpose(np.dot(np.transpose(dL_dscores), X) / N)  # D x 1
        grad_b = np.sum(dL_dscores) / N

        w += -param_delta * grad_w
        b += -param_delta * grad_b

    return w, b


def binlogreg_classify(X, w, b):
    """
      Argumenti
          X:    podatci, np.array NxD
          w, b: parametri logističke regresije

      Povratne vrijednosti
          probs: vjerojatnosti razreda c1
    """
    scores = np.dot(X, w) + b
    exp_s = np.exp(scores)
    probs = exp_s / (1 + exp_s)
    return probs


def binlogreg_decfun(w, b):
    return lambda X: binlogreg_classify(X, w, b)


if __name__ == "__main__":
    np.random.seed(100)

    # get the training dataset
    X, Y_ = sample_gauss_2d(2, 100)
    Y_ = np.reshape(Y_, (-1, 1))

    # train the model
    w, b = binlogreg_train(X, Y_)

    # evaluate the model on the training dataset
    probs = binlogreg_classify(X, w, b)
    Y = (probs >= 0.5).astype(int)

    probs = np.reshape(probs, (-1, ))
    Y = np.reshape(Y, (-1, ))
    Y_ = np.reshape(Y, (-1, ))

    # report perforamce
    accuracy, recall, precision = eval_perf_binary(Y, Y_)
    AP = eval_AP(Y_[probs.argsort()])
    print(accuracy, recall, precision, AP)

    # graph the decision surface
    decfun = binlogreg_decfun(w, b)
    bbox = (np.min(X, axis=0), np.max(X, axis=0))
    graph_surface(decfun, bbox, offset=0.5)

    # graph the data points
    graph_data(X, Y_, Y, special=[])

    # show the plot
    plt.show()
