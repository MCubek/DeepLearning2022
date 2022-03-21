import data
import numpy as np
import matplotlib.pyplot as plt

ReLU = lambda x: np.maximum(0, x)


def fcann2_forward(X, W1, b1, W2, b2):
    h1 = ReLU(np.dot(X, W1.T) + b1)  # N x H
    s2 = np.dot(h1, W2.T) + b2  # N x C

    exp_s2 = np.exp(s2)  # N x C
    exp_s2_sum = np.sum(exp_s2, axis=1, keepdims=True)

    prob = exp_s2 / (1 + exp_s2_sum)  # N x C

    return h1, prob


def fcann2_train(X, Y_, param_niter=1e5, param_delta=0.05, param_lambda=1e-3, param_nhidden=5):
    N, D = X.shape
    C = np.unique(Y_).size

    # X -> N X D

    W1 = np.random.randn(param_nhidden, D)  # H x D
    b1 = np.zeros((1, param_nhidden))  # 1 x H
    W2 = np.random.randn(C, param_nhidden)  # C x H
    b2 = np.zeros((1, C))  # 1 x C

    for iter_n in range(int(param_niter)):
        h1, prob = fcann2_forward(X, W1, b1, W2, b2)

        log_prob = -np.log(prob[range(N), Y_])

        regularization = param_lambda * (np.sum(np.square(W1)) + np.sum(np.square(W2)))

        log_loss = np.sum(log_prob) / N + regularization

        if iter_n % 500 == 0:
            print(f"Iter {iter_n}, loss {log_loss}.")

        Gs2 = prob  # N x C
        Gs2[range(N), Y_] -= 1
        Gs2 /= N

        grad_W2 = np.dot(Gs2.T, h1)
        grad_b2 = np.sum(Gs2, axis=0, keepdims=True)  # C x 1

        Gs1 = np.dot(Gs2, W2)  # N x H
        Gs1[h1 <= 0] = 0

        grad_W1 = np.dot(Gs1.T, X)  # H x D
        grad_b1 = np.sum(Gs1, axis=0, keepdims=True)  # H x 1

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
