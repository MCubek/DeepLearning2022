import data
import numpy as np
from sklearn.svm import SVC
import matplotlib.pyplot as plt


class KSVMWrap:
    """
    Metode:
      __init__(self, X, Y_, param_svm_c=1, param_svm_gamma='auto'):
        Konstruira omotač i uči RBF SVM klasifikator
        X, Y_:           podatci i točni indeksi razreda
        param_svm_c:     relativni značaj podatkovne cijene
        param_svm_gamma: širina RBF jezgre

      predict(self, X)
        Predviđa i vraća indekse razreda podataka X

      get_scores(self, X):
        Vraća klasifikacijske mjere
        (engl. classification scores) podataka X;
        ovo će vam trebati za računanje prosječne preciznosti.

      support
        Indeksi podataka koji su odabrani za potporne vektore
    """

    def __init__(self, X, Y_, param_svm_c=1, param_svm_gamma='auto'):
        self.svc = SVC(C=param_svm_c, gamma=param_svm_gamma)
        self.svc.fit(X, Y_)

    def predict(self, X):
        return self.svc.predict(X)

    def get_scores(self, X):
        return self.svc.decision_function(X)

    def support(self):
        return self.svc.support_


if __name__ == '__main__':
    np.random.seed(100)

    X, Y_ = data.sample_gmm_2d(6, 2, 10)

    svm = KSVMWrap(X, Y_)

    y_predicted = svm.predict(X)

    acc, recall, precission = data.eval_perf_binary(y_predicted, Y_)
    print(f'accuracy:{acc}\nrecall :{recall}\nprecission:{precission}')

    # iscrtaj rezultate, decizijsku plohu
    rect = (np.min(X, axis=0), np.max(X, axis=0))
    data.graph_surface(lambda x: svm.predict(x), rect, offset=0)

    # graph the data points
    data.graph_data(X, Y_, y_predicted, special=[svm.support()])

    plt.show()
