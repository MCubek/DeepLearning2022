import data
import numpy as np
import matplotlib.pyplot as plt

ReLU = lambda x: np.maximum(0, x)

softmax = lambda x: np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)


def loss(prob, N, y, W1, W2, param_lambda):
    log_loss = - np.sum(np.log(prob[range(N), y])) / N
    regularization = param_lambda * (np.sum(np.square(W1)) + np.sum(np.square(W2)))
    return log_loss + regularization


def fcann2_forward(X, W1, b1, W2, b2):
    h1 = ReLU(np.dot(X, W1) + b1)  # N x H
    s2 = np.dot(h1, W2) + b2  # N x C

    prob = softmax(s2)  # N x C

    return h1, prob


def fcann2_train(X, Y_, param_niter=1e5, param_delta=0.05, param_lambda=1e-3, param_nhidden=5):
    N, D = X.shape
    C = np.unique(Y_).size

    # X -> N X D
    W1 = np.random.randn(D, param_nhidden)  # D x H
    b1 = np.zeros((1, param_nhidden))  # 1 x H
    W2 = np.random.randn(param_nhidden, C)  # H x C
    b2 = np.zeros((1, C))  # 1 x C

    for iter_n in range(int(param_niter)):
        h1, prob = fcann2_forward(X, W1, b1, W2, b2)

        log_loss = loss(prob, N, Y_, W1, W2, param_lambda)

        if iter_n % 500 == 0:
            print(f"Iter {iter_n}, loss {log_loss}.")

        Gs2 = prob  # N x C
        Gs2[range(N), Y_] -= 1
        Gs2 /= N

        Gs1 = np.dot(Gs2, W2.T)  # N x H
        Gs1[h1 <= 0] = 0

        grad_W2 = np.dot(h1.T, Gs2)
        grad_b2 = np.sum(Gs2, axis=0, keepdims=True)  # C x 1

        grad_W1 = np.dot(X.T, Gs1)  # H x D
        grad_b1 = np.sum(Gs1, axis=0, keepdims=True)  # H x 1

        grad_W2 += param_lambda * W2
        grad_W1 += param_lambda * W1

        W2 += - param_delta * grad_W2
        b2 += - param_delta * grad_b2
        W1 += - param_delta * grad_W1
        b1 += - param_delta * grad_b1

    return W1, b1, W2, b2


classify = lambda x: fcann2_forward(x, W1, b1, W2, b2)[1]

if __name__ == '__main__':
    np.random.seed(100)

    # Generate data
    X, Y_ = data.sample_gmm_2d(6, 2, 10)

    # Train model
    W1, b1, W2, b2 = fcann2_train(X, Y_)

    # Get predictions
    y_predicted = np.argmax(classify(X), axis=1)

    # Graph
    rect = (np.min(X, axis=0), np.max(X, axis=0))
    data.graph_surface(lambda x: classify(x)[:, 0], rect, offset=0.45)
    data.graph_data(X, Y_, y_predicted, special=[])

    plt.show()
