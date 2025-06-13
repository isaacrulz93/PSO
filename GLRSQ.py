import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from pyriemann.utils.mean import mean_riemann
from pyriemann.utils.distance import distance_riemann
from scipy.linalg import sqrtm, inv, logm, expm


class GLRSQ(BaseEstimator, ClassifierMixin):
    """
    Riemannian GLRSQ with proper gradient descent updates on SPD manifold.
    Implements E(Pi) = logistic((dK - dJ) / (dK + dJ))
    using Riemannian gradient directions from the paper.
    """
    def __init__(self, n_classes=4, n_prototypes=1, max_iter=5, learning_rate=0.1):
        self.n_classes = n_classes
        self.n_prototypes = n_prototypes
        self.max_iter = max_iter
        self.lr = learning_rate

    def fit(self, X, y):
        self.prototypes_ = []
        self.labels_ = []
        classes = np.unique(y)

        for cls in classes:
            Xc = X[y == cls]
            proto = mean_riemann(Xc)
            self.prototypes_.append(proto)
            self.labels_.append(cls)

        self.prototypes_ = np.stack(self.prototypes_)
        self.labels_ = np.array(self.labels_)

        for _ in range(self.max_iter):
            for Pi, yi in zip(X, y):
                dists = np.array([distance_riemann(Pi, W) for W in self.prototypes_])
                same_mask = (self.labels_ == yi)
                diff_mask = ~same_mask

                idx_J = np.argmin(dists[same_mask])
                idx_K = np.argmin(dists[diff_mask])
                J = np.where(same_mask)[0][idx_J]
                K = np.where(diff_mask)[0][idx_K]

                WJ = self.prototypes_[J]
                WK = self.prototypes_[K]

                dJ = dists[J] + 1e-8
                dK = dists[K] + 1e-8

                omega = 1 / (1 + np.exp(-(dK - dJ) / (dK + dJ)))
                coeff_J = -omega * 4 * dK / (dJ + dK)**2
                coeff_K =  omega * 4 * dJ / (dJ + dK)**2

                grad_J = self._logmap(WJ, Pi)
                grad_K = self._logmap(WK, Pi)

                self.prototypes_[J] = self._expmap(WJ, -self.lr * coeff_J * grad_J)
                self.prototypes_[K] = self._expmap(WK, -self.lr * coeff_K * grad_K)

        return self

    def predict(self, X):
        preds = []
        for Pi in X:
            dists = np.array([distance_riemann(Pi, W) for W in self.prototypes_])
            pred = self.labels_[np.argmin(dists)]
            preds.append(pred)
        return np.array(preds)

    @staticmethod
    def _logmap(base, P):
        base_sqrt = sqrtm(base)
        base_inv_sqrt = inv(base_sqrt)
        return base_sqrt @ logm(base_inv_sqrt @ P @ base_inv_sqrt) @ base_sqrt

    @staticmethod
    def _expmap(base, V):
        base_sqrt = sqrtm(base)
        base_inv_sqrt = inv(base_sqrt)
        return base_sqrt @ expm(base_inv_sqrt @ V @ base_inv_sqrt) @ base_sqrt